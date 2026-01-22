from collections import namedtuple, OrderedDict
import gymnasium as gym
import logging
import re
import tree  # pip install dm_tree
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from ray.util.debug import log_once
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import (
@DeveloperAPI
class DynamicTFPolicy(TFPolicy):
    """A TFPolicy that auto-defines placeholders dynamically at runtime.

    Do not sub-class this class directly (neither should you sub-class
    TFPolicy), but rather use rllib.policy.tf_policy_template.build_tf_policy
    to generate your custom tf (graph-mode or eager) Policy classes.
    """

    @DeveloperAPI
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict, loss_fn: Callable[[Policy, ModelV2, Type[TFActionDistribution], SampleBatch], TensorType], *, stats_fn: Optional[Callable[[Policy, SampleBatch], Dict[str, TensorType]]]=None, grad_stats_fn: Optional[Callable[[Policy, SampleBatch, ModelGradients], Dict[str, TensorType]]]=None, before_loss_init: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], None]]=None, make_model: Optional[Callable[[Policy, gym.spaces.Space, gym.spaces.Space, AlgorithmConfigDict], ModelV2]]=None, action_sampler_fn: Optional[Callable[[TensorType, List[TensorType]], Union[Tuple[TensorType, TensorType], Tuple[TensorType, TensorType, TensorType, List[TensorType]]]]]=None, action_distribution_fn: Optional[Callable[[Policy, ModelV2, TensorType, TensorType, TensorType], Tuple[TensorType, type, List[TensorType]]]]=None, existing_inputs: Optional[Dict[str, 'tf1.placeholder']]=None, existing_model: Optional[ModelV2]=None, get_batch_divisibility_req: Optional[Callable[[Policy], int]]=None, obs_include_prev_action_reward=DEPRECATED_VALUE):
        """Initializes a DynamicTFPolicy instance.

        Initialization of this class occurs in two phases and defines the
        static graph.

        Phase 1: The model is created and model variables are initialized.

        Phase 2: A fake batch of data is created, sent to the trajectory
        postprocessor, and then used to create placeholders for the loss
        function. The loss and stats functions are initialized with these
        placeholders.

        Args:
            observation_space: Observation space of the policy.
            action_space: Action space of the policy.
            config: Policy-specific configuration data.
            loss_fn: Function that returns a loss tensor for the policy graph.
            stats_fn: Optional callable that - given the policy and batch
                input tensors - returns a dict mapping str to TF ops.
                These ops are fetched from the graph after loss calculations
                and the resulting values can be found in the results dict
                returned by e.g. `Algorithm.train()` or in tensorboard (if TB
                logging is enabled).
            grad_stats_fn: Optional callable that - given the policy, batch
                input tensors, and calculated loss gradient tensors - returns
                a dict mapping str to TF ops. These ops are fetched from the
                graph after loss and gradient calculations and the resulting
                values can be found in the results dict returned by e.g.
                `Algorithm.train()` or in tensorboard (if TB logging is
                enabled).
            before_loss_init: Optional function to run prior to
                loss init that takes the same arguments as __init__.
            make_model: Optional function that returns a ModelV2 object
                given policy, obs_space, action_space, and policy config.
                All policy variables should be created in this function. If not
                specified, a default model will be created.
            action_sampler_fn: A callable returning either a sampled action and
                its log-likelihood or a sampled action, its log-likelihood,
                action distribution inputs and updated state given Policy,
                ModelV2, observation inputs, explore, and is_training.
                Provide `action_sampler_fn` if you would like to have full
                control over the action computation step, including the
                model forward pass, possible sampling from a distribution,
                and exploration logic.
                Note: If `action_sampler_fn` is given, `action_distribution_fn`
                must be None. If both `action_sampler_fn` and
                `action_distribution_fn` are None, RLlib will simply pass
                inputs through `self.model` to get distribution inputs, create
                the distribution object, sample from it, and apply some
                exploration logic to the results.
                The callable takes as inputs: Policy, ModelV2, obs_batch,
                state_batches (optional), seq_lens (optional),
                prev_actions_batch (optional), prev_rewards_batch (optional),
                explore, and is_training.
            action_distribution_fn: A callable returning distribution inputs
                (parameters), a dist-class to generate an action distribution
                object from, and internal-state outputs (or an empty list if
                not applicable).
                Provide `action_distribution_fn` if you would like to only
                customize the model forward pass call. The resulting
                distribution parameters are then used by RLlib to create a
                distribution object, sample from it, and execute any
                exploration logic.
                Note: If `action_distribution_fn` is given, `action_sampler_fn`
                must be None. If both `action_sampler_fn` and
                `action_distribution_fn` are None, RLlib will simply pass
                inputs through `self.model` to get distribution inputs, create
                the distribution object, sample from it, and apply some
                exploration logic to the results.
                The callable takes as inputs: Policy, ModelV2, input_dict,
                explore, timestep, is_training.
            existing_inputs: When copying a policy, this specifies an existing
                dict of placeholders to use instead of defining new ones.
            existing_model: When copying a policy, this specifies an existing
                model to clone and share weights with.
            get_batch_divisibility_req: Optional callable that returns the
                divisibility requirement for sample batches. If None, will
                assume a value of 1.
        """
        if obs_include_prev_action_reward != DEPRECATED_VALUE:
            deprecation_warning(old='obs_include_prev_action_reward', error=True)
        self.observation_space = obs_space
        self.action_space = action_space
        self.config = config
        self.framework = 'tf'
        self._loss_fn = loss_fn
        self._stats_fn = stats_fn
        self._grad_stats_fn = grad_stats_fn
        self._seq_lens = None
        self._is_tower = existing_inputs is not None
        dist_class = None
        if action_sampler_fn or action_distribution_fn:
            if not make_model:
                raise ValueError('`make_model` is required if `action_sampler_fn` OR `action_distribution_fn` is given')
        else:
            dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, self.config['model'])
        if existing_model:
            if isinstance(existing_model, list):
                self.model = existing_model[0]
                for i in range(1, len(existing_model)):
                    setattr(self, existing_model[i][0], existing_model[i][1])
        elif make_model:
            self.model = make_model(self, obs_space, action_space, config)
        else:
            self.model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=logit_dim, model_config=self.config['model'], framework='tf')
        self._update_model_view_requirements_from_init_state()
        if existing_inputs:
            self._state_inputs = [v for k, v in existing_inputs.items() if k.startswith('state_in_')]
            if self._state_inputs:
                self._seq_lens = existing_inputs[SampleBatch.SEQ_LENS]
        else:
            self._state_inputs = [get_placeholder(space=vr.space, time_axis=not isinstance(vr.shift, int), name=k) for k, vr in self.model.view_requirements.items() if k.startswith('state_in_')]
            if self._state_inputs:
                self._seq_lens = tf1.placeholder(dtype=tf.int32, shape=[None], name='seq_lens')
        self.view_requirements = self._get_default_view_requirements()
        self.view_requirements.update(self.model.view_requirements)
        if SampleBatch.INFOS in self.view_requirements:
            self.view_requirements[SampleBatch.INFOS].used_for_training = False
        if self._is_tower:
            timestep = existing_inputs['timestep']
            explore = False
            self._input_dict, self._dummy_batch = self._get_input_dict_and_dummy_batch(self.view_requirements, existing_inputs)
        else:
            if not self.config.get('_disable_action_flattening'):
                action_ph = ModelCatalog.get_action_placeholder(action_space)
                prev_action_ph = {}
                if SampleBatch.PREV_ACTIONS not in self.view_requirements:
                    prev_action_ph = {SampleBatch.PREV_ACTIONS: ModelCatalog.get_action_placeholder(action_space, 'prev_action')}
                self._input_dict, self._dummy_batch = self._get_input_dict_and_dummy_batch(self.view_requirements, dict({SampleBatch.ACTIONS: action_ph}, **prev_action_ph))
            else:
                self._input_dict, self._dummy_batch = self._get_input_dict_and_dummy_batch(self.view_requirements, {})
            timestep = tf1.placeholder_with_default(tf.zeros((), dtype=tf.int64), (), name='timestep')
            explore = tf1.placeholder_with_default(True, (), name='is_exploring')
        self._input_dict.set_training(self._get_is_training_placeholder())
        sampled_action = None
        sampled_action_logp = None
        dist_inputs = None
        extra_action_fetches = {}
        self._state_out = None
        if not self._is_tower:
            self.exploration = self._create_exploration()
            if action_sampler_fn:
                action_sampler_outputs = action_sampler_fn(self, self.model, obs_batch=self._input_dict[SampleBatch.CUR_OBS], state_batches=self._state_inputs, seq_lens=self._seq_lens, prev_action_batch=self._input_dict.get(SampleBatch.PREV_ACTIONS), prev_reward_batch=self._input_dict.get(SampleBatch.PREV_REWARDS), explore=explore, is_training=self._input_dict.is_training)
                if len(action_sampler_outputs) == 4:
                    sampled_action, sampled_action_logp, dist_inputs, self._state_out = action_sampler_outputs
                else:
                    dist_inputs = None
                    self._state_out = []
                    sampled_action, sampled_action_logp = action_sampler_outputs
            else:
                if action_distribution_fn:
                    in_dict = self._input_dict
                    try:
                        dist_inputs, dist_class, self._state_out = action_distribution_fn(self, self.model, input_dict=in_dict, state_batches=self._state_inputs, seq_lens=self._seq_lens, explore=explore, timestep=timestep, is_training=in_dict.is_training)
                    except TypeError as e:
                        if 'positional argument' in e.args[0] or 'unexpected keyword argument' in e.args[0]:
                            dist_inputs, dist_class, self._state_out = action_distribution_fn(self, self.model, obs_batch=in_dict[SampleBatch.CUR_OBS], state_batches=self._state_inputs, seq_lens=self._seq_lens, prev_action_batch=in_dict.get(SampleBatch.PREV_ACTIONS), prev_reward_batch=in_dict.get(SampleBatch.PREV_REWARDS), explore=explore, is_training=in_dict.is_training)
                        else:
                            raise e
                elif isinstance(self.model, tf.keras.Model):
                    dist_inputs, self._state_out, extra_action_fetches = self.model(self._input_dict)
                else:
                    dist_inputs, self._state_out = self.model(self._input_dict)
                action_dist = dist_class(dist_inputs, self.model)
                sampled_action, sampled_action_logp = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
        if dist_inputs is not None:
            extra_action_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
        if sampled_action_logp is not None:
            extra_action_fetches[SampleBatch.ACTION_LOGP] = sampled_action_logp
            extra_action_fetches[SampleBatch.ACTION_PROB] = tf.exp(tf.cast(sampled_action_logp, tf.float32))
        sess = tf1.get_default_session() or tf1.Session(config=tf1.ConfigProto(**self.config['tf_session_args']))
        batch_divisibility_req = get_batch_divisibility_req(self) if callable(get_batch_divisibility_req) else get_batch_divisibility_req or 1
        prev_action_input = self._input_dict[SampleBatch.PREV_ACTIONS] if SampleBatch.PREV_ACTIONS in self._input_dict.accessed_keys else None
        prev_reward_input = self._input_dict[SampleBatch.PREV_REWARDS] if SampleBatch.PREV_REWARDS in self._input_dict.accessed_keys else None
        super().__init__(observation_space=obs_space, action_space=action_space, config=config, sess=sess, obs_input=self._input_dict[SampleBatch.OBS], action_input=self._input_dict[SampleBatch.ACTIONS], sampled_action=sampled_action, sampled_action_logp=sampled_action_logp, dist_inputs=dist_inputs, dist_class=dist_class, loss=None, loss_inputs=[], model=self.model, state_inputs=self._state_inputs, state_outputs=self._state_out, prev_action_input=prev_action_input, prev_reward_input=prev_reward_input, seq_lens=self._seq_lens, max_seq_len=config['model']['max_seq_len'], batch_divisibility_req=batch_divisibility_req, explore=explore, timestep=timestep)
        if before_loss_init is not None:
            before_loss_init(self, obs_space, action_space, config)
        if hasattr(self, '_extra_action_fetches'):
            self._extra_action_fetches.update(extra_action_fetches)
        else:
            self._extra_action_fetches = extra_action_fetches
        if not self._is_tower:
            self._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs=True)
            if len(self.devices) > 1 or any(('gpu' in d for d in self.devices)):
                with tf1.variable_scope('', reuse=tf1.AUTO_REUSE):
                    self.multi_gpu_tower_stacks = [TFMultiGPUTowerStack(policy=self) for i in range(self.config.get('num_multi_gpu_tower_stacks', 1))]
            self.get_session().run(tf1.global_variables_initializer())

    @override(TFPolicy)
    @DeveloperAPI
    def copy(self, existing_inputs: List[Tuple[str, 'tf1.placeholder']]) -> TFPolicy:
        """Creates a copy of self using existing input placeholders."""
        flat_loss_inputs = tree.flatten(self._loss_input_dict)
        flat_loss_inputs_no_rnn = tree.flatten(self._loss_input_dict_no_rnn)
        if len(flat_loss_inputs) != len(existing_inputs):
            raise ValueError('Tensor list mismatch', self._loss_input_dict, self._state_inputs, existing_inputs)
        for i, v in enumerate(flat_loss_inputs_no_rnn):
            if v.shape.as_list() != existing_inputs[i].shape.as_list():
                raise ValueError('Tensor shape mismatch', i, v.shape, existing_inputs[i].shape)
        rnn_inputs = []
        for i in range(len(self._state_inputs)):
            rnn_inputs.append(('state_in_{}'.format(i), existing_inputs[len(flat_loss_inputs_no_rnn) + i]))
        if rnn_inputs:
            rnn_inputs.append((SampleBatch.SEQ_LENS, existing_inputs[-1]))
        existing_inputs_unflattened = tree.unflatten_as(self._loss_input_dict_no_rnn, existing_inputs[:len(flat_loss_inputs_no_rnn)])
        input_dict = OrderedDict([('is_exploring', self._is_exploring), ('timestep', self._timestep)] + [(k, existing_inputs_unflattened[k]) for i, k in enumerate(self._loss_input_dict_no_rnn.keys())] + rnn_inputs)
        instance = self.__class__(self.observation_space, self.action_space, self.config, existing_inputs=input_dict, existing_model=[self.model, ('target_q_model', getattr(self, 'target_q_model', None)), ('target_model', getattr(self, 'target_model', None))])
        instance._loss_input_dict = input_dict
        losses = instance._do_loss_init(SampleBatch(input_dict))
        loss_inputs = [(k, existing_inputs_unflattened[k]) for i, k in enumerate(self._loss_input_dict_no_rnn.keys())]
        TFPolicy._initialize_loss(instance, losses, loss_inputs)
        if instance._grad_stats_fn:
            instance._stats_fetches.update(instance._grad_stats_fn(instance, input_dict, instance._grads))
        return instance

    @override(Policy)
    @DeveloperAPI
    def get_initial_state(self) -> List[TensorType]:
        if self.model:
            return self.model.get_initial_state()
        else:
            return []

    @override(Policy)
    @DeveloperAPI
    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int=0) -> int:
        batch.set_training(True)
        if len(self.devices) == 1 and self.devices[0] == '/cpu:0':
            assert buffer_index == 0
            self._loaded_single_cpu_batch = batch
            return len(batch)
        input_dict = self._get_loss_inputs_dict(batch, shuffle=False)
        data_keys = tree.flatten(self._loss_input_dict_no_rnn)
        if self._state_inputs:
            state_keys = self._state_inputs + [self._seq_lens]
        else:
            state_keys = []
        inputs = [input_dict[k] for k in data_keys]
        state_inputs = [input_dict[k] for k in state_keys]
        return self.multi_gpu_tower_stacks[buffer_index].load_data(sess=self.get_session(), inputs=inputs, state_inputs=state_inputs, num_grad_updates=batch.num_grad_updates)

    @override(Policy)
    @DeveloperAPI
    def get_num_samples_loaded_into_buffer(self, buffer_index: int=0) -> int:
        if len(self.devices) == 1 and self.devices[0] == '/cpu:0':
            assert buffer_index == 0
            return len(self._loaded_single_cpu_batch) if self._loaded_single_cpu_batch is not None else 0
        return self.multi_gpu_tower_stacks[buffer_index].num_tuples_loaded

    @override(Policy)
    @DeveloperAPI
    def learn_on_loaded_batch(self, offset: int=0, buffer_index: int=0):
        if len(self.devices) == 1 and self.devices[0] == '/cpu:0':
            assert buffer_index == 0
            if self._loaded_single_cpu_batch is None:
                raise ValueError('Must call Policy.load_batch_into_buffer() before Policy.learn_on_loaded_batch()!')
            batch_size = self.config.get('sgd_minibatch_size', self.config['train_batch_size'])
            if batch_size >= len(self._loaded_single_cpu_batch):
                sliced_batch = self._loaded_single_cpu_batch
            else:
                sliced_batch = self._loaded_single_cpu_batch.slice(start=offset, end=offset + batch_size)
            return self.learn_on_batch(sliced_batch)
        tower_stack = self.multi_gpu_tower_stacks[buffer_index]
        results = tower_stack.optimize(self.get_session(), offset)
        self.num_grad_updates += 1
        results.update({NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (tower_stack.num_grad_updates or 0)})
        return results

    def _get_input_dict_and_dummy_batch(self, view_requirements, existing_inputs):
        """Creates input_dict and dummy_batch for loss initialization.

        Used for managing the Policy's input placeholders and for loss
        initialization.
        Input_dict: Str -> tf.placeholders, dummy_batch: str -> np.arrays.

        Args:
            view_requirements: The view requirements dict.
            existing_inputs (Dict[str, tf.placeholder]): A dict of already
                existing placeholders.

        Returns:
            Tuple[Dict[str, tf.placeholder], Dict[str, np.ndarray]]: The
                input_dict/dummy_batch tuple.
        """
        input_dict = {}
        for view_col, view_req in view_requirements.items():
            mo = re.match('state_in_(\\d+)', view_col)
            if mo is not None:
                input_dict[view_col] = self._state_inputs[int(mo.group(1))]
            elif view_col.startswith('state_out_'):
                continue
            elif view_col == SampleBatch.ACTION_DIST_INPUTS:
                continue
            elif view_col in existing_inputs:
                input_dict[view_col] = existing_inputs[view_col]
            else:
                time_axis = not isinstance(view_req.shift, int)
                if view_req.used_for_training:
                    if self.config.get('_disable_action_flattening') and view_col in [SampleBatch.ACTIONS, SampleBatch.PREV_ACTIONS]:
                        flatten = False
                    elif view_col in [SampleBatch.OBS, SampleBatch.NEXT_OBS] and self.config['_disable_preprocessor_api']:
                        flatten = False
                    else:
                        flatten = True
                    input_dict[view_col] = get_placeholder(space=view_req.space, name=view_col, time_axis=time_axis, flatten=flatten)
        dummy_batch = self._get_dummy_batch_from_view_requirements(batch_size=32)
        return (SampleBatch(input_dict, seq_lens=self._seq_lens), dummy_batch)

    @override(Policy)
    def _initialize_loss_from_dummy_batch(self, auto_remove_unneeded_view_reqs: bool=True, stats_fn=None) -> None:
        if not self._optimizers:
            self._optimizers = force_list(self.optimizer())
            self._optimizer = self._optimizers[0]
        self.get_session().run(tf1.global_variables_initializer())
        for key, view_req in self.view_requirements.items():
            if not key.startswith('state_in_') and key not in self._input_dict.accessed_keys:
                view_req.used_for_compute_actions = False
        for key, value in self._extra_action_fetches.items():
            self._dummy_batch[key] = get_dummy_batch_for_space(gym.spaces.Box(-1.0, 1.0, shape=value.shape.as_list()[1:], dtype=value.dtype.name), batch_size=len(self._dummy_batch))
            self._input_dict[key] = get_placeholder(value=value, name=key)
            if key not in self.view_requirements:
                logger.info('Adding extra-action-fetch `{}` to view-reqs.'.format(key))
                self.view_requirements[key] = ViewRequirement(space=gym.spaces.Box(-1.0, 1.0, shape=value.shape.as_list()[1:], dtype=value.dtype.name), used_for_compute_actions=False)
        dummy_batch = self._dummy_batch
        logger.info('Testing `postprocess_trajectory` w/ dummy batch.')
        self.exploration.postprocess_trajectory(self, dummy_batch, self.get_session())
        _ = self.postprocess_trajectory(dummy_batch)
        for key in dummy_batch.added_keys:
            if key not in self._input_dict:
                self._input_dict[key] = get_placeholder(value=dummy_batch[key], name=key)
            if key not in self.view_requirements:
                self.view_requirements[key] = ViewRequirement(space=gym.spaces.Box(-1.0, 1.0, shape=dummy_batch[key].shape[1:], dtype=dummy_batch[key].dtype), used_for_compute_actions=False)
        train_batch = SampleBatch(dict(self._input_dict, **self._loss_input_dict), _is_training=True)
        if self._state_inputs:
            train_batch[SampleBatch.SEQ_LENS] = self._seq_lens
            self._loss_input_dict.update({SampleBatch.SEQ_LENS: train_batch[SampleBatch.SEQ_LENS]})
        self._loss_input_dict.update({k: v for k, v in train_batch.items()})
        if log_once('loss_init'):
            logger.debug('Initializing loss function with dummy input:\n\n{}\n'.format(summarize(train_batch)))
        losses = self._do_loss_init(train_batch)
        all_accessed_keys = train_batch.accessed_keys | dummy_batch.accessed_keys | dummy_batch.added_keys | set(self.model.view_requirements.keys())
        TFPolicy._initialize_loss(self, losses, [(k, v) for k, v in train_batch.items() if k in all_accessed_keys] + ([(SampleBatch.SEQ_LENS, train_batch[SampleBatch.SEQ_LENS])] if SampleBatch.SEQ_LENS in train_batch else []))
        if 'is_training' in self._loss_input_dict:
            del self._loss_input_dict['is_training']
        if self._grad_stats_fn:
            self._stats_fetches.update(self._grad_stats_fn(self, train_batch, self._grads))
        if auto_remove_unneeded_view_reqs:
            all_accessed_keys = train_batch.accessed_keys | dummy_batch.accessed_keys
            for key in dummy_batch.accessed_keys:
                if key not in train_batch.accessed_keys and key not in self.model.view_requirements and (key not in [SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.UNROLL_ID, SampleBatch.TERMINATEDS, SampleBatch.TRUNCATEDS, SampleBatch.REWARDS, SampleBatch.INFOS, SampleBatch.T, SampleBatch.OBS_EMBEDS]):
                    if key in self.view_requirements:
                        self.view_requirements[key].used_for_training = False
                    if key in self._loss_input_dict:
                        del self._loss_input_dict[key]
            for key in list(self.view_requirements.keys()):
                if key not in all_accessed_keys and key not in [SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.UNROLL_ID, SampleBatch.TERMINATEDS, SampleBatch.TRUNCATEDS, SampleBatch.REWARDS, SampleBatch.INFOS, SampleBatch.T] and (key not in self.model.view_requirements):
                    if key in dummy_batch.deleted_keys:
                        logger.warning("SampleBatch key '{}' was deleted manually in postprocessing function! RLlib will automatically remove non-used items from the data stream. Remove the `del` from your postprocessing function.".format(key))
                    elif self.config['output'] is None:
                        del self.view_requirements[key]
                    if key in self._loss_input_dict:
                        del self._loss_input_dict[key]
            for key in list(self.view_requirements.keys()):
                vr = self.view_requirements[key]
                if vr.data_col is not None and vr.data_col not in self.view_requirements:
                    used_for_training = vr.data_col in train_batch.accessed_keys
                    self.view_requirements[vr.data_col] = ViewRequirement(space=vr.space, used_for_training=used_for_training)
        self._loss_input_dict_no_rnn = {k: v for k, v in self._loss_input_dict.items() if v not in self._state_inputs and v != self._seq_lens}

    def _do_loss_init(self, train_batch: SampleBatch):
        losses = self._loss_fn(self, self.model, self.dist_class, train_batch)
        losses = force_list(losses)
        if self._stats_fn:
            self._stats_fetches.update(self._stats_fn(self, train_batch))
        self._update_ops = []
        if not isinstance(self.model, tf.keras.Model):
            self._update_ops = self.model.update_ops()
        return losses