import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.init import constant_
from xformers.components.attention import Attention
from xformers.components.input_projection import InputProjection, InputProjectionConfig
from xformers.components.positional_embedding import RotaryEmbedding

        Expected input dimensions are [batch size, sequence length, embed dim]
        Output dimensions are [batch size, sequence length, embed dim]
        