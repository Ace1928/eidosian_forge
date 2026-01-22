from gymnasium.spaces import Box
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
class TorchParametricActionsModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, true_obs_shape=(4,), action_embed_size=2, **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = TorchFC(Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size, model_config, name + '_action_embed')

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict['obs']['avail_actions']
        action_mask = input_dict['obs']['action_mask']
        action_embed, _ = self.action_embed_model({'obs': input_dict['obs']['cart']})
        intent_vector = torch.unsqueeze(action_embed, 1)
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return (action_logits + inf_mask, state)

    def value_function(self):
        return self.action_embed_model.value_function()