from gymnasium.spaces import Box
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
class ParametricActionsModel(DistributionalQTFModel):
    """Parametric action model that handles the dot product and masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(4,), action_embed_size=2, **kw):
        super(ParametricActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size, model_config, name + '_action_embed')

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict['obs']['avail_actions']
        action_mask = input_dict['obs']['action_mask']
        action_embed, _ = self.action_embed_model({'obs': input_dict['obs']['cart']})
        intent_vector = tf.expand_dims(action_embed, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return (action_logits + inf_mask, state)

    def value_function(self):
        return self.action_embed_model.value_function()