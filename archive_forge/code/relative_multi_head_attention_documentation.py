from typing import Union
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.utils.typing import TensorType
Initializes a RelativeMultiHeadAttention nn.Module object.

        Args:
            in_dim (int):
            out_dim: The output dimension of this module. Also known as
                "attention dim".
            num_heads: The number of attention heads to use.
                Denoted `H` in [2].
            head_dim: The dimension of a single(!) attention head
                Denoted `D` in [2].
            input_layernorm: Whether to prepend a LayerNorm before
                everything else. Should be True for building a GTrXL.
            output_activation (Union[str, callable]): Optional activation
                function or activation function specifier (str).
                Should be "relu" for GTrXL.
            **kwargs:
        