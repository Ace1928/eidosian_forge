from gymnasium.spaces import Discrete, Tuple
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import normc_initializer as normc_init_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class _ActionModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.a2_hidden = SlimFC(in_size=1, out_size=16, activation_fn=nn.Tanh, initializer=normc_init_torch(1.0))
        self.a2_logits = SlimFC(in_size=16, out_size=2, activation_fn=None, initializer=normc_init_torch(0.01))

    def forward(self_, ctx_input, a1_input):
        a1_logits = self.a1_logits(ctx_input)
        a2_logits = self_.a2_logits(self_.a2_hidden(a1_input))
        return (a1_logits, a2_logits)