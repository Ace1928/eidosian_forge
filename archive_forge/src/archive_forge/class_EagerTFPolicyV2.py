import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.eager_tf_policy import (
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import (
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@DeveloperAPI
class EagerTFPolicyV2(Policy):
    """A TF-eager / TF2 based tensorflow policy.

    This class is intended to be used and extended by sub-classing.
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict, **kwargs):
        self.framework = config.get('framework', 'tf2')
        logger.info('Creating TF-eager policy running on {}.'.format('GPU' if get_gpu_devices() else 'CPU'))
        Policy.__init__(self, observation_space, action_space, config)
        self._is_training = False
        self.global_timestep = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.explore = tf.Variable(self.config['explore'], trainable=False, dtype=tf.bool)
        num_gpus = self._get_num_gpus_for_policy()
        if num_gpus > 0:
            gpu_ids = get_gpu_devices()
            logger.info(f'Found {len(gpu_ids)} visible cuda devices.')
        self._is_training = False
        self._loss_initialized = False
        self._loss = None
        self.batch_divisibility_req = self.get_batch_divisibility_req()
        self._max_seq_len = self.config['model']['max_seq_len']
        self.validate_spaces(observation_space, action_space, self.config)
        if self.config.get('_enable_new_api_stack', False):
            self.model = self.make_rl_module()
            self.dist_class = None
        else:
            self.dist_class = self._init_dist_class()
            self.model = self.make_model()
        self._init_view_requirements()
        if self.config.get('_enable_new_api_stack', False):
            self.exploration = None
        else:
            self.exploration = self._create_exploration()
        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        self.global_timestep.assign(0)
        self._lock = threading.RLock()
        self._re_trace_counter = 0

    @DeveloperAPI
    @staticmethod
    def enable_eager_execution_if_necessary():
        if tf1 and (not tf1.executing_eagerly()):
            tf1.enable_eager_execution()

    @ExperimentalAPI
    @override(Policy)
    def maybe_remove_time_dimension(self, input_dict: Dict[str, TensorType]):
        assert self.config.get('_enable_new_api_stack', False), 'This is a helper method for the new learner API.'
        if self.config.get('_enable_new_api_stack', False) and self.model.is_stateful():
            ret = {}

            def fold_mapping(item):
                item = tf.convert_to_tensor(item)
                shape = tf.shape(item)
                b_dim, t_dim = (shape[0], shape[1])
                other_dims = shape[2:]
                return tf.reshape(item, tf.concat([[b_dim * t_dim], other_dims], axis=0))
            for k, v in input_dict.items():
                if k not in (STATE_IN, STATE_OUT):
                    ret[k] = tree.map_structure(fold_mapping, v)
                else:
                    ret[k] = v
            return ret
        else:
            return input_dict

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def validate_spaces(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict):
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    @override(Policy)
    def loss(self, model: Union[ModelV2, 'tf.keras.Model'], dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        """Compute loss for this policy using model, dist_class and a train_batch.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            A single loss tensor or a list of loss tensors.
        """
        if self.config.get('_enable_new_api_stack', False):
            for k in model.input_specs_train():
                train_batch[k]
            return None
        else:
            raise NotImplementedError

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Stats function. Returns a dict of statistics.

        Args:
            train_batch: The SampleBatch (already) used for training.

        Returns:
            The stats dict.
        """
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def grad_stats_fn(self, train_batch: SampleBatch, grads: ModelGradients) -> Dict[str, TensorType]:
        """Gradient stats function. Returns a dict of statistics.

        Args:
            train_batch: The SampleBatch (already) used for training.

        Returns:
            The stats dict.
        """
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def make_model(self) -> ModelV2:
        """Build underlying model for this Policy.

        Returns:
            The Model for the Policy to use.
        """
        _, logit_dim = ModelCatalog.get_action_dist(self.action_space, self.config['model'])
        return ModelCatalog.get_model_v2(self.observation_space, self.action_space, logit_dim, self.config['model'], framework=self.framework)

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def compute_gradients_fn(self, policy: Policy, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
        """Gradients computing function (from loss tensor, using local optimizer).

        Args:
            policy: The Policy object that generated the loss tensor and
                that holds the given local optimizer.
            optimizer: The tf (local) optimizer object to
                calculate the gradients with.
            loss: The loss tensor for which gradients should be
                calculated.

        Returns:
            ModelGradients: List of the possibly clipped gradients- and variable
                tuples.
        """
        return None

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def apply_gradients_fn(self, optimizer: 'tf.keras.optimizers.Optimizer', grads: ModelGradients) -> 'tf.Operation':
        """Gradients computing function (from loss tensor, using local optimizer).

        Args:
            optimizer: The tf (local) optimizer object to
                calculate the gradients with.
            grads: The gradient tensor to be applied.

        Returns:
            "tf.Operation": TF operation that applies supplied gradients.
        """
        return None

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def action_sampler_fn(self, model: ModelV2, *, obs_batch: TensorType, state_batches: TensorType, **kwargs) -> Tuple[TensorType, TensorType, TensorType, List[TensorType]]:
        """Custom function for sampling new actions given policy.

        Args:
            model: Underlying model.
            obs_batch: Observation tensor batch.
            state_batches: Action sampling state batch.

        Returns:
            Sampled action
            Log-likelihood
            Action distribution inputs
            Updated state
        """
        return (None, None, None, None)

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def action_distribution_fn(self, model: ModelV2, *, obs_batch: TensorType, state_batches: TensorType, **kwargs) -> Tuple[TensorType, type, List[TensorType]]:
        """Action distribution function for this Policy.

        Args:
            model: Underlying model.
            obs_batch: Observation tensor batch.
            state_batches: Action sampling state batch.

        Returns:
            Distribution input.
            ActionDistribution class.
            State outs.
        """
        return (None, None, None)

    @DeveloperAPI
    @OverrideToImplementCustomLogic
    def get_batch_divisibility_req(self) -> int:
        """Get batch divisibility request.

        Returns:
            Size N. A sample batch must be of size K*N.
        """
        return 1

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_action_out_fn(self) -> Dict[str, TensorType]:
        """Extra values to fetch and return from compute_actions().

        Returns:
             Dict[str, TensorType]: An extra fetch-dict to be passed to and
                returned from the compute_actions() call.
        """
        return {}

    @DeveloperAPI
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def extra_learn_fetches_fn(self) -> Dict[str, TensorType]:
        """Extra stats to be reported after gradient computation.

        Returns:
             Dict[str, TensorType]: An extra fetch-dict.
        """
        return {}

    @override(Policy)
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[SampleBatch]=None, episode: Optional['Episode']=None):
        """Post process trajectory in the format of a SampleBatch.

        Args:
            sample_batch: sample_batch: batch of experiences for the policy,
                which will contain at most one episode trajectory.
            other_agent_batches: In a multi-agent env, this contains a
                mapping of agent ids to (policy, agent_batch) tuples
                containing the policy and experiences of the other agents.
            episode: An optional multi-agent episode object to provide
                access to all of the internal episode state, which may
                be useful for model-based or multi-agent algorithms.

        Returns:
            The postprocessed sample batch.
        """
        assert tf.executing_eagerly()
        return Policy.postprocess_trajectory(self, sample_batch)

    @OverrideToImplementCustomLogic
    def optimizer(self) -> Union['tf.keras.optimizers.Optimizer', List['tf.keras.optimizers.Optimizer']]:
        """TF optimizer to use for policy optimization.

        Returns:
            A local optimizer or a list of local optimizers to use for this
                Policy's Model.
        """
        return tf.keras.optimizers.Adam(self.config['lr'])

    def _init_dist_class(self):
        if is_overridden(self.action_sampler_fn) or is_overridden(self.action_distribution_fn):
            if not is_overridden(self.make_model):
                raise ValueError('`make_model` is required if `action_sampler_fn` OR `action_distribution_fn` is given')
            return None
        else:
            dist_class, _ = ModelCatalog.get_action_dist(self.action_space, self.config['model'])
            return dist_class

    def _init_view_requirements(self):
        if self.config.get('_enable_new_api_stack', False):
            self.view_requirements = self.model.update_default_view_requirements(self.view_requirements)
        else:
            self._update_model_view_requirements_from_init_state()
            self.view_requirements.update(self.model.view_requirements)
        if SampleBatch.INFOS in self.view_requirements:
            self.view_requirements[SampleBatch.INFOS].used_for_training = False

    def maybe_initialize_optimizer_and_loss(self):
        if not self.config.get('_enable_new_api_stack', False):
            optimizers = force_list(self.optimizer())
            if self.exploration:
                optimizers = self.exploration.get_exploration_optimizer(optimizers)
            self._optimizers: List[LocalOptimizer] = optimizers
            self._optimizer: LocalOptimizer = optimizers[0] if optimizers else None
        self._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs=True)
        self._loss_initialized = True

    @override(Policy)
    def compute_actions_from_input_dict(self, input_dict: Dict[str, TensorType], explore: bool=None, timestep: Optional[int]=None, episodes: Optional[List[Episode]]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        self._is_training = False
        explore = explore if explore is not None else self.explore
        timestep = timestep if timestep is not None else self.global_timestep
        if isinstance(timestep, tf.Tensor):
            timestep = int(timestep.numpy())
        input_dict = self._lazy_tensor_dict(input_dict)
        input_dict.set_training(False)
        state_batches = [input_dict[k] for k in input_dict.keys() if 'state_in' in k[:8]]
        self._state_in = state_batches
        self._is_recurrent = len(tree.flatten(self._state_in)) > 0
        if self.exploration:
            self.exploration.before_compute_actions(timestep=timestep, explore=explore, tf_sess=self.get_session())
        if self.config.get('_enable_new_api_stack'):
            seq_lens = input_dict.get('seq_lens', None)
            if seq_lens is None:
                if not isinstance(input_dict, SampleBatch):
                    input_dict = SampleBatch(input_dict)
                seq_lens = np.array([1] * len(input_dict))
            input_dict = self.maybe_add_time_dimension(input_dict, seq_lens=seq_lens)
            if explore:
                ret = self._compute_actions_helper_rl_module_explore(input_dict)
            else:
                ret = self._compute_actions_helper_rl_module_inference(input_dict)
        else:
            ret = self._compute_actions_helper(input_dict, state_batches, None if self.config['eager_tracing'] else episodes, explore, timestep)
        self.global_timestep.assign_add(tree.flatten(ret[0])[0].shape.as_list()[0])
        return convert_to_numpy(ret)

    @override(Policy)
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, explore=None, timestep=None, **kwargs):
        input_dict = SampleBatch({SampleBatch.CUR_OBS: obs_batch}, _is_training=tf.constant(False))
        if state_batches is not None:
            for s in enumerate(state_batches):
                input_dict['state_in_{i}'] = s
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        if info_batch is not None:
            input_dict[SampleBatch.INFOS] = info_batch
        return self.compute_actions_from_input_dict(input_dict=input_dict, explore=explore, timestep=timestep, episodes=episodes, **kwargs)

    @with_lock
    @override(Policy)
    def compute_log_likelihoods(self, actions: Union[List[TensorType], TensorType], obs_batch: Union[List[TensorType], TensorType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Optional[Union[List[TensorType], TensorType]]=None, prev_reward_batch: Optional[Union[List[TensorType], TensorType]]=None, actions_normalized: bool=True, in_training: bool=True) -> TensorType:
        if is_overridden(self.action_sampler_fn) and (not is_overridden(self.action_distribution_fn)):
            raise ValueError('Cannot compute log-prob/likelihood w/o an `action_distribution_fn` and a provided `action_sampler_fn`!')
        seq_lens = tf.ones(len(obs_batch), dtype=tf.int32)
        input_batch = SampleBatch({SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_batch), SampleBatch.ACTIONS: actions}, _is_training=False)
        if prev_action_batch is not None:
            input_batch[SampleBatch.PREV_ACTIONS] = tf.convert_to_tensor(prev_action_batch)
        if prev_reward_batch is not None:
            input_batch[SampleBatch.PREV_REWARDS] = tf.convert_to_tensor(prev_reward_batch)
        if self.exploration:
            self.exploration.before_compute_actions(explore=False)
        if is_overridden(self.action_distribution_fn):
            dist_inputs, self.dist_class, _ = self.action_distribution_fn(self, self.model, input_batch, explore=False, is_training=False)
            action_dist = self.dist_class(dist_inputs, self.model)
        elif self.config.get('_enable_new_api_stack', False):
            if in_training:
                output = self.model.forward_train(input_batch)
                action_dist_cls = self.model.get_train_action_dist_cls()
                if action_dist_cls is None:
                    raise ValueError('The RLModules must provide an appropriate action distribution class for training if is_eval_mode is False.')
            else:
                output = self.model.forward_exploration(input_batch)
                action_dist_cls = self.model.get_exploration_action_dist_cls()
                if action_dist_cls is None:
                    raise ValueError('The RLModules must provide an appropriate action distribution class for exploration if is_eval_mode is True.')
            action_dist_inputs = output.get(SampleBatch.ACTION_DIST_INPUTS, None)
            if action_dist_inputs is None:
                raise ValueError('The RLModules must provide inputs to create the action distribution. These should be part of the output of the appropriate forward method under the key SampleBatch.ACTION_DIST_INPUTS.')
            action_dist = action_dist_cls.from_logits(action_dist_inputs)
        else:
            dist_inputs, _ = self.model(input_batch, state_batches, seq_lens)
            action_dist = self.dist_class(dist_inputs, self.model)
        if not actions_normalized and self.config['normalize_actions']:
            actions = normalize_action(actions, self.action_space_struct)
        log_likelihoods = action_dist.logp(actions)
        return log_likelihoods

    @with_lock
    @override(Policy)
    def learn_on_batch(self, postprocessed_batch):
        learn_stats = {}
        self.callbacks.on_learn_on_batch(policy=self, train_batch=postprocessed_batch, result=learn_stats)
        pad_batch_to_sequences_of_same_size(postprocessed_batch, max_seq_len=self._max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
        self._is_training = True
        postprocessed_batch = self._lazy_tensor_dict(postprocessed_batch)
        postprocessed_batch.set_training(True)
        stats = self._learn_on_batch_helper(postprocessed_batch)
        self.num_grad_updates += 1
        stats.update({'custom_metrics': learn_stats, NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count, NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (postprocessed_batch.num_grad_updates or 0)})
        return convert_to_numpy(stats)

    @override(Policy)
    def compute_gradients(self, postprocessed_batch: SampleBatch) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        pad_batch_to_sequences_of_same_size(postprocessed_batch, shuffle=False, max_seq_len=self._max_seq_len, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
        self._is_training = True
        self._lazy_tensor_dict(postprocessed_batch)
        postprocessed_batch.set_training(True)
        grads_and_vars, grads, stats = self._compute_gradients_helper(postprocessed_batch)
        return convert_to_numpy((grads, stats))

    @override(Policy)
    def apply_gradients(self, gradients: ModelGradients) -> None:
        self._apply_gradients_helper(list(zip([tf.convert_to_tensor(g) if g is not None else None for g in gradients], self.model.trainable_variables())))

    @override(Policy)
    def get_weights(self, as_dict=False):
        variables = self.variables()
        if as_dict:
            return {v.name: v.numpy() for v in variables}
        return [v.numpy() for v in variables]

    @override(Policy)
    def set_weights(self, weights):
        variables = self.variables()
        assert len(weights) == len(variables), (len(weights), len(variables))
        for v, w in zip(variables, weights):
            v.assign(w)

    @override(Policy)
    def get_exploration_state(self):
        return convert_to_numpy(self.exploration.get_state())

    @override(Policy)
    def is_recurrent(self):
        return self._is_recurrent

    @override(Policy)
    def num_state_tensors(self):
        return len(self._state_inputs)

    @override(Policy)
    def get_initial_state(self):
        if hasattr(self, 'model'):
            return self.model.get_initial_state()
        return []

    @override(Policy)
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def get_state(self) -> PolicyState:
        state = super().get_state()
        state['global_timestep'] = state['global_timestep'].numpy()
        state['_optimizer_variables'] = []
        if not self.config.get('_enable_new_api_stack', False):
            if self._optimizer and len(self._optimizer.variables()) > 0:
                state['_optimizer_variables'] = self._optimizer.variables()
        if self.exploration:
            state['_exploration_state'] = self.exploration.get_state()
        return state

    @override(Policy)
    @OverrideToImplementCustomLogic_CallToSuperRecommended
    def set_state(self, state: PolicyState) -> None:
        optimizer_vars = state.get('_optimizer_variables', None)
        if optimizer_vars and self._optimizer.variables():
            if not type(self).__name__.endswith('_traced') and log_once('set_state_optimizer_vars_tf_eager_policy_v2'):
                logger.warning("Cannot restore an optimizer's state for tf eager! Keras is not able to save the v1.x optimizers (from tf.compat.v1.train) since they aren't compatible with checkpoints.")
            for opt_var, value in zip(self._optimizer.variables(), optimizer_vars):
                opt_var.assign(value)
        if hasattr(self, 'exploration') and '_exploration_state' in state:
            self.exploration.set_state(state=state['_exploration_state'])
        self.global_timestep.assign(state['global_timestep'])
        super().set_state(state)

    @override(Policy)
    def export_model(self, export_dir, onnx: Optional[int]=None) -> None:
        enable_rl_module_api = self.config.get('enable_rl_module_api', False)
        if enable_rl_module_api:
            raise ValueError('ONNX export not supported for RLModule API.')
        if onnx:
            try:
                import tf2onnx
            except ImportError as e:
                raise RuntimeError('Converting a TensorFlow model to ONNX requires `tf2onnx` to be installed. Install with `pip install tf2onnx`.') from e
            model_proto, external_tensor_storage = tf2onnx.convert.from_keras(self.model.base_model, output_path=os.path.join(export_dir, 'model.onnx'))
        elif hasattr(self, 'model') and hasattr(self.model, 'base_model') and isinstance(self.model.base_model, tf.keras.Model):
            try:
                self.model.base_model.save(export_dir, save_format='tf')
            except Exception:
                logger.warning(ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL)
        else:
            logger.warning(ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL)

    def variables(self):
        """Return the list of all savable variables for this policy."""
        if isinstance(self.model, tf.keras.Model):
            return self.model.variables
        else:
            return self.model.variables()

    def loss_initialized(self):
        return self._loss_initialized

    @with_lock
    def _compute_actions_helper_rl_module_explore(self, input_dict, _ray_trace_ctx=None):
        self._re_trace_counter += 1
        extra_fetches = {}
        input_dict = NestedDict(input_dict)
        fwd_out = self.model.forward_exploration(input_dict)
        fwd_out = self.maybe_remove_time_dimension(fwd_out)
        action_dist = None
        if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
            action_dist_class = self.model.get_exploration_action_dist_cls()
            action_dist = action_dist_class.from_logits(fwd_out[SampleBatch.ACTION_DIST_INPUTS])
        if SampleBatch.ACTIONS in fwd_out:
            actions = fwd_out[SampleBatch.ACTIONS]
        else:
            if action_dist is None:
                raise KeyError(f"Your RLModule's `forward_exploration()` method must return a dictwith either the {SampleBatch.ACTIONS} key or the {SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!")
            actions = action_dist.sample()
        for k, v in fwd_out.items():
            if k not in [SampleBatch.ACTIONS, 'state_out']:
                extra_fetches[k] = v
        if action_dist is not None:
            logp = action_dist.logp(actions)
            extra_fetches[SampleBatch.ACTION_LOGP] = logp
            extra_fetches[SampleBatch.ACTION_PROB] = tf.exp(logp)
        state_out = convert_to_numpy(fwd_out.get('state_out', {}))
        return (actions, state_out, extra_fetches)

    @with_lock
    def _compute_actions_helper_rl_module_inference(self, input_dict, _ray_trace_ctx=None):
        self._re_trace_counter += 1
        extra_fetches = {}
        input_dict = NestedDict(input_dict)
        fwd_out = self.model.forward_inference(input_dict)
        fwd_out = self.maybe_remove_time_dimension(fwd_out)
        action_dist = None
        if SampleBatch.ACTION_DIST_INPUTS in fwd_out:
            action_dist_class = self.model.get_inference_action_dist_cls()
            action_dist = action_dist_class.from_logits(fwd_out[SampleBatch.ACTION_DIST_INPUTS])
            action_dist = action_dist.to_deterministic()
        if SampleBatch.ACTIONS in fwd_out:
            actions = fwd_out[SampleBatch.ACTIONS]
        else:
            if action_dist is None:
                raise KeyError(f"Your RLModule's `forward_inference()` method must return a dictwith either the {SampleBatch.ACTIONS} key or the {SampleBatch.ACTION_DIST_INPUTS} key in it (or both)!")
            actions = action_dist.sample()
        for k, v in fwd_out.items():
            if k not in [SampleBatch.ACTIONS, 'state_out']:
                extra_fetches[k] = v
        state_out = convert_to_numpy(fwd_out.get('state_out', {}))
        return (actions, state_out, extra_fetches)

    @with_lock
    def _compute_actions_helper(self, input_dict, state_batches, episodes, explore, timestep, _ray_trace_ctx=None):
        self._re_trace_counter += 1
        if SampleBatch.SEQ_LENS in input_dict:
            seq_lens = input_dict[SampleBatch.SEQ_LENS]
        else:
            batch_size = tree.flatten(input_dict[SampleBatch.OBS])[0].shape[0]
            seq_lens = tf.ones(batch_size, dtype=tf.int32) if state_batches else None
        extra_fetches = {}
        with tf.variable_creator_scope(_disallow_var_creation):
            if is_overridden(self.action_sampler_fn):
                actions, logp, dist_inputs, state_out = self.action_sampler_fn(self.model, input_dict[SampleBatch.OBS], explore=explore, timestep=timestep, episodes=episodes)
            else:
                if is_overridden(self.action_distribution_fn):
                    dist_inputs, self.dist_class, state_out = self.action_distribution_fn(self.model, obs_batch=input_dict[SampleBatch.OBS], state_batches=state_batches, seq_lens=seq_lens, explore=explore, timestep=timestep, is_training=False)
                elif isinstance(self.model, tf.keras.Model):
                    if state_batches and 'state_in_0' not in input_dict:
                        for i, s in enumerate(state_batches):
                            input_dict[f'state_in_{i}'] = s
                    self._lazy_tensor_dict(input_dict)
                    dist_inputs, state_out, extra_fetches = self.model(input_dict)
                else:
                    dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)
                action_dist = self.dist_class(dist_inputs, self.model)
                actions, logp = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = tf.exp(logp)
            extra_fetches[SampleBatch.ACTION_LOGP] = logp
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
        extra_fetches.update(self.extra_action_out_fn())
        return (actions, state_out, extra_fetches)

    def _learn_on_batch_helper(self, samples, _ray_trace_ctx=None):
        self._re_trace_counter += 1
        with tf.variable_creator_scope(_disallow_var_creation):
            grads_and_vars, _, stats = self._compute_gradients_helper(samples)
        self._apply_gradients_helper(grads_and_vars)
        return stats

    def _get_is_training_placeholder(self):
        return tf.convert_to_tensor(self._is_training)

    @with_lock
    def _compute_gradients_helper(self, samples):
        """Computes and returns grads as eager tensors."""
        self._re_trace_counter += 1
        if isinstance(self.model, tf.keras.Model):
            variables = self.model.trainable_variables
        else:
            variables = self.model.trainable_variables()
        with tf.GradientTape(persistent=is_overridden(self.compute_gradients_fn)) as tape:
            losses = self.loss(self.model, self.dist_class, samples)
        losses = force_list(losses)
        if is_overridden(self.compute_gradients_fn):
            optimizer = _OptimizerWrapper(tape)
            if self.config['_tf_policy_handles_more_than_one_loss']:
                grads_and_vars = self.compute_gradients_fn([optimizer] * len(losses), losses)
            else:
                grads_and_vars = [self.compute_gradients_fn(optimizer, losses[0])]
        else:
            grads_and_vars = [list(zip(tape.gradient(loss, variables), variables)) for loss in losses]
        if log_once('grad_vars'):
            for g_and_v in grads_and_vars:
                for g, v in g_and_v:
                    if g is not None:
                        logger.info(f'Optimizing variable {v.name}')
        if self.config['_tf_policy_handles_more_than_one_loss']:
            grads = [[g for g, _ in g_and_v] for g_and_v in grads_and_vars]
        else:
            grads_and_vars = grads_and_vars[0]
            grads = [g for g, _ in grads_and_vars]
        stats = self._stats(samples, grads)
        return (grads_and_vars, grads, stats)

    def _apply_gradients_helper(self, grads_and_vars):
        self._re_trace_counter += 1
        if is_overridden(self.apply_gradients_fn):
            if self.config['_tf_policy_handles_more_than_one_loss']:
                self.apply_gradients_fn(self._optimizers, grads_and_vars)
            else:
                self.apply_gradients_fn(self._optimizer, grads_and_vars)
        elif self.config['_tf_policy_handles_more_than_one_loss']:
            for i, o in enumerate(self._optimizers):
                o.apply_gradients([(g, v) for g, v in grads_and_vars[i] if g is not None])
        else:
            self._optimizer.apply_gradients([(g, v) for g, v in grads_and_vars if g is not None])

    def _stats(self, samples, grads):
        fetches = {}
        if is_overridden(self.stats_fn):
            fetches[LEARNER_STATS_KEY] = {k: v for k, v in self.stats_fn(samples).items()}
        else:
            fetches[LEARNER_STATS_KEY] = {}
        fetches.update({k: v for k, v in self.extra_learn_fetches_fn().items()})
        fetches.update({k: v for k, v in self.grad_stats_fn(samples, grads).items()})
        return fetches

    def _lazy_tensor_dict(self, postprocessed_batch: SampleBatch):
        if not isinstance(postprocessed_batch, SampleBatch):
            postprocessed_batch = SampleBatch(postprocessed_batch)
        postprocessed_batch.set_get_interceptor(_convert_to_tf)
        return postprocessed_batch

    @classmethod
    def with_tracing(cls):
        return _traced_eager_policy(cls)