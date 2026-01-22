from gymnasium.spaces import Dict
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        orig_space = getattr(obs_space, 'original_space', obs_space)
        assert isinstance(orig_space, Dict) and 'action_mask' in orig_space.spaces and ('observations' in orig_space.spaces)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)
        self.internal_model = TorchFC(orig_space['observations'], action_space, num_outputs, model_config, name + '_internal')
        self.no_masking = False
        if 'no_masking' in model_config['custom_model_config']:
            self.no_masking = model_config['custom_model_config']['no_masking']

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict['obs']['action_mask']
        logits, _ = self.internal_model({'obs': input_dict['obs']['observations']})
        if self.no_masking:
            return (logits, state)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        return (masked_logits, state)

    def value_function(self):
        return self.internal_model.value_function()