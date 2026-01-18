from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.utils.framework import TensorType

        in_dim: Dimension of input
        out_dim: Dimension of output
        num_heads: Number of attention heads
        head_dim: Output dimension of each attention head
        