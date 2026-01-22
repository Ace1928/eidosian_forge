from gymnasium.spaces import Box
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import (
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class ContActionQModel(TFModelV2):
    """A simple, q-value-from-cont-action model (for e.g. SAC type algos)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ContActionQModel, self).__init__(obs_space, action_space, None, model_config, name)
        combined_space = Box(-1.0, 1.0, (self.num_outputs + action_space.shape[0],))
        self.q_head = FullyConnectedNetwork(combined_space, action_space, 1, model_config, 'q_head')

    def get_single_q_value(self, underlying_output, action):
        input_ = tf.concat([underlying_output, action], axis=-1)
        input_dict = {'obs': input_}
        q_values, _ = self.q_head(input_dict)
        return q_values