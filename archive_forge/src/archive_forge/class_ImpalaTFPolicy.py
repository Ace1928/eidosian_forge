import numpy as np
import logging
import gymnasium as gym
from typing import Dict, List, Optional, Type, Union
from ray.rllib.algorithms.impala import vtrace_tf as vtrace
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_bootstrap_value
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.policy.tf_mixins import GradStatsMixin, ValueNetworkMixin
from ray.rllib.utils.typing import (
class ImpalaTFPolicy(VTraceClipGradients, VTraceOptimizer, LearningRateSchedule, EntropyCoeffSchedule, GradStatsMixin, ValueNetworkMixin, base):

    def __init__(self, observation_space, action_space, config, existing_model=None, existing_inputs=None):
        base.enable_eager_execution_if_necessary()
        base.__init__(self, observation_space, action_space, config, existing_inputs=existing_inputs, existing_model=existing_model)
        ValueNetworkMixin.__init__(self, config)
        if not self.config.get('_enable_new_api_stack'):
            GradStatsMixin.__init__(self)
            VTraceClipGradients.__init__(self)
            VTraceOptimizer.__init__(self)
            LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
            EntropyCoeffSchedule.__init__(self, config['entropy_coeff'], config['entropy_coeff_schedule'])
        self.maybe_initialize_optimizer_and_loss()

    @override(base)
    def loss(self, model: Union[ModelV2, 'tf.keras.Model'], dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)
        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def make_time_major(*args, **kw):
            return _make_time_major(self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kw)
        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
        unpacked_behaviour_logits = tf.split(behaviour_logits, output_hidden_shape, axis=1)
        unpacked_outputs = tf.split(model_out, output_hidden_shape, axis=1)
        values = model.value_function()
        values_time_major = make_time_major(values)
        bootstrap_values_time_major = make_time_major(train_batch[SampleBatch.VALUES_BOOTSTRAPPED])
        bootstrap_value = bootstrap_values_time_major[-1]
        if self.is_recurrent():
            max_seq_len = tf.reduce_max(train_batch[SampleBatch.SEQ_LENS])
            mask = tf.sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = tf.reshape(mask, [-1])
        else:
            mask = tf.ones_like(rewards)
        loss_actions = actions if is_multidiscrete else tf.expand_dims(actions, axis=1)
        self.vtrace_loss = VTraceLoss(actions=make_time_major(loss_actions), actions_logp=make_time_major(action_dist.logp(actions)), actions_entropy=make_time_major(action_dist.multi_entropy()), dones=make_time_major(dones), behaviour_action_logp=make_time_major(behaviour_action_logp), behaviour_logits=make_time_major(unpacked_behaviour_logits), target_logits=make_time_major(unpacked_outputs), discount=self.config['gamma'], rewards=make_time_major(rewards), values=values_time_major, bootstrap_value=bootstrap_value, dist_class=Categorical if is_multidiscrete else dist_class, model=model, valid_mask=make_time_major(mask), config=self.config, vf_loss_coeff=self.config['vf_loss_coeff'], entropy_coeff=self.entropy_coeff, clip_rho_threshold=self.config['vtrace_clip_rho_threshold'], clip_pg_rho_threshold=self.config['vtrace_clip_pg_rho_threshold'])
        if self.config.get('_separate_vf_optimizer'):
            return (self.vtrace_loss.loss_wo_vf, self.vtrace_loss.vf_loss)
        else:
            return self.vtrace_loss.total_loss

    @override(base)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        values_batched = _make_time_major(self, train_batch.get(SampleBatch.SEQ_LENS), self.model.value_function())
        return {'cur_lr': tf.cast(self.cur_lr, tf.float64), 'policy_loss': self.vtrace_loss.mean_pi_loss, 'entropy': self.vtrace_loss.mean_entropy, 'entropy_coeff': tf.cast(self.entropy_coeff, tf.float64), 'var_gnorm': tf.linalg.global_norm(self.model.trainable_variables()), 'vf_loss': self.vtrace_loss.mean_vf_loss, 'vf_explained_var': explained_variance(tf.reshape(self.vtrace_loss.value_targets, [-1]), tf.reshape(values_batched, [-1]))}

    @override(base)
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[SampleBatch]=None, episode: Optional['Episode']=None):
        if self.config['vtrace']:
            sample_batch = compute_bootstrap_value(sample_batch, self)
        return sample_batch

    @override(base)
    def get_batch_divisibility_req(self) -> int:
        return self.config['rollout_fragment_length']