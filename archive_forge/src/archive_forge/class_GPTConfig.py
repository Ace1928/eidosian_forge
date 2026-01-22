import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
@DeveloperAPI
@dataclass
class GPTConfig:
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    embed_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1