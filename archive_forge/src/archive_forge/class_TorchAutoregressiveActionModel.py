from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class TorchAutoregressiveActionModel(TorchModelV2, nn.Module):
    """PyTorch version of the AutoregressiveActionModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        if action_space != Tuple([Discrete(2), Discrete(2)]):
            raise ValueError('This model only supports the [2, 2] action space')
        self.context_layer = SlimFC(in_size=obs_space.shape[0], out_size=num_outputs, initializer=normc_init_torch(1.0), activation_fn=nn.Tanh)
        self.value_branch = SlimFC(in_size=num_outputs, out_size=1, initializer=normc_init_torch(0.01), activation_fn=None)
        self.a1_logits = SlimFC(in_size=num_outputs, out_size=2, activation_fn=None, initializer=normc_init_torch(0.01))

        class _ActionModel(nn.Module):

            def __init__(self):
                nn.Module.__init__(self)
                self.a2_hidden = SlimFC(in_size=1, out_size=16, activation_fn=nn.Tanh, initializer=normc_init_torch(1.0))
                self.a2_logits = SlimFC(in_size=16, out_size=2, activation_fn=None, initializer=normc_init_torch(0.01))

            def forward(self_, ctx_input, a1_input):
                a1_logits = self.a1_logits(ctx_input)
                a2_logits = self_.a2_logits(self_.a2_hidden(a1_input))
                return (a1_logits, a2_logits)
        self.action_module = _ActionModel()
        self._context = None

    def forward(self, input_dict, state, seq_lens):
        self._context = self.context_layer(input_dict['obs'])
        return (self._context, state)

    def value_function(self):
        return torch.reshape(self.value_branch(self._context), [-1])