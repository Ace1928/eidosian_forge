import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
from xformers.components.attention.utils import (

        key_padding_mask    Only a key padding mask is accepted here. The size must be (batch size, sequence length) or
                            (batch size * num_heads, 1, sequence length). If dimensions are not correct, the mask will
                            be ignored. An additive mask is expected, meaning float values using "-inf" to mask values
        