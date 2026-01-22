import tensorflow as tf
from typing import Any, Mapping
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.tf.tf_distributions import TfCategorical
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
class BCTfRLModuleWithSharedGlobalEncoder(TfRLModule):

    def __init__(self, encoder, local_dim, hidden_dim, action_dim):
        super().__init__()
        self.encoder = encoder
        self.policy_head = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim + local_dim, input_shape=(hidden_dim + local_dim,), activation='relu'), tf.keras.layers.Dense(hidden_dim, activation='relu'), tf.keras.layers.Dense(action_dim)])

    @override(RLModule)
    def _default_input_specs(self):
        return [('obs', 'global'), ('obs', 'local')]

    @override(RLModule)
    def _forward_inference(self, batch):
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch):
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        obs = batch['obs']
        global_enc = self.encoder(obs['global'])
        policy_in = tf.concat([global_enc, obs['local']], axis=-1)
        action_logits = self.policy_head(policy_in)
        return {SampleBatch.ACTION_DIST_INPUTS: action_logits}