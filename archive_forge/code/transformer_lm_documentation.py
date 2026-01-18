import math
import torch
import torch.nn as nn
from fairscale.nn.moe.moe_layer import MOELayer
from fairscale.nn.moe.top2gate import Top2Gate
Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        