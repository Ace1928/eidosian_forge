from typing import (
import gymnasium as gym
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils import add_mixins, NullContextManager
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import try_import_torch, try_import_jax
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ModelGradients, TensorType, AlgorithmConfigDict
class policy_cls(base):

    def __init__(self, obs_space, action_space, config):
        self.config = config
        self.framework = self.config['framework'] = framework
        if validate_spaces:
            validate_spaces(self, obs_space, action_space, self.config)
        if before_init:
            before_init(self, obs_space, action_space, self.config)
        if make_model:
            assert make_model_and_action_dist is None, 'Either `make_model` or `make_model_and_action_dist` must be None!'
            self.model = make_model(self, obs_space, action_space, config)
            dist_class, _ = ModelCatalog.get_action_dist(action_space, self.config['model'], framework=framework)
        elif make_model_and_action_dist:
            self.model, dist_class = make_model_and_action_dist(self, obs_space, action_space, config)
        else:
            dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, self.config['model'], framework=framework)
            self.model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=logit_dim, model_config=self.config['model'], framework=framework)
        model_cls = TorchModelV2
        assert isinstance(self.model, model_cls), 'ERROR: Generated Model must be a TorchModelV2 object!'
        self.parent_cls = parent_cls
        self.parent_cls.__init__(self, observation_space=obs_space, action_space=action_space, config=config, model=self.model, loss=None if self.config['in_evaluation'] else loss_fn, action_distribution_class=dist_class, action_sampler_fn=action_sampler_fn, action_distribution_fn=action_distribution_fn, max_seq_len=config['model']['max_seq_len'], get_batch_divisibility_req=get_batch_divisibility_req)
        self.view_requirements.update(self.model.view_requirements)
        _before_loss_init = before_loss_init or after_init
        if _before_loss_init:
            _before_loss_init(self, self.observation_space, self.action_space, config)
        self._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs=True, stats_fn=None if self.config['in_evaluation'] else stats_fn)
        if _after_loss_init:
            _after_loss_init(self, obs_space, action_space, config)
        self.global_timestep = 0

    @override(Policy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        with self._no_grad_context():
            sample_batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
            if postprocess_fn:
                return postprocess_fn(self, sample_batch, other_agent_batches, episode)
            return sample_batch

    @override(parent_cls)
    def extra_grad_process(self, optimizer, loss):
        """Called after optimizer.zero_grad() and loss.backward() calls.

            Allows for gradient processing before optimizer.step() is called.
            E.g. for gradient clipping.
            """
        if extra_grad_process_fn:
            return extra_grad_process_fn(self, optimizer, loss)
        else:
            return parent_cls.extra_grad_process(self, optimizer, loss)

    @override(parent_cls)
    def extra_compute_grad_fetches(self):
        if extra_learn_fetches_fn:
            fetches = convert_to_numpy(extra_learn_fetches_fn(self))
            return dict({LEARNER_STATS_KEY: {}}, **fetches)
        else:
            return parent_cls.extra_compute_grad_fetches(self)

    @override(parent_cls)
    def compute_gradients(self, batch):
        if compute_gradients_fn:
            return compute_gradients_fn(self, batch)
        else:
            return parent_cls.compute_gradients(self, batch)

    @override(parent_cls)
    def apply_gradients(self, gradients):
        if apply_gradients_fn:
            apply_gradients_fn(self, gradients)
        else:
            parent_cls.apply_gradients(self, gradients)

    @override(parent_cls)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        with self._no_grad_context():
            if extra_action_out_fn:
                stats_dict = extra_action_out_fn(self, input_dict, state_batches, model, action_dist)
            else:
                stats_dict = parent_cls.extra_action_out(self, input_dict, state_batches, model, action_dist)
            return self._convert_to_numpy(stats_dict)

    @override(parent_cls)
    def optimizer(self):
        if optimizer_fn:
            optimizers = optimizer_fn(self, self.config)
        else:
            optimizers = parent_cls.optimizer(self)
        return optimizers

    @override(parent_cls)
    def extra_grad_info(self, train_batch):
        with self._no_grad_context():
            if stats_fn:
                stats_dict = stats_fn(self, train_batch)
            else:
                stats_dict = self.parent_cls.extra_grad_info(self, train_batch)
            return self._convert_to_numpy(stats_dict)

    def _no_grad_context(self):
        if self.framework == 'torch':
            return torch.no_grad()
        return NullContextManager()

    def _convert_to_numpy(self, data):
        if self.framework == 'torch':
            return convert_to_numpy(data)
        return data