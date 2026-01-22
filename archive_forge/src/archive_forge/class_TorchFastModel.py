from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class TorchFastModel(TorchModelV2, nn.Module):
    """Torch version of FastModel (tf)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, requires_grad=True))
        self.dummy_layer = SlimFC(1, 1)
        self._output = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._output = self.bias + torch.zeros(size=(input_dict['obs'].shape[0], self.num_outputs)).to(self.bias.device)
        return (self._output, [])

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, 'must call forward first!'
        return torch.reshape(torch.mean(self._output, -1), [-1])