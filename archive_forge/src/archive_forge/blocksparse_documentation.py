import logging
import math
from dataclasses import dataclass
import torch
from xformers import _is_triton_available
from xformers.components.attention import Attention, AttentionConfig, register_attention

            A thin wrap around the Triton blockparse attention operation

            .. note: Per element attention mask is not supported, but you can specify causality
            