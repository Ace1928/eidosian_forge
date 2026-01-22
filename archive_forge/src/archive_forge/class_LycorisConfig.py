from __future__ import annotations
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from peft.config import PeftConfig
from peft.utils import (
from .tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge, check_target_module_exists
@dataclass
class LycorisConfig(PeftConfig):
    """
    A base config for LyCORIS like adapters
    """
    rank_pattern: Optional[dict] = field(default_factory=dict, metadata={'help': 'The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}'})
    alpha_pattern: Optional[dict] = field(default_factory=dict, metadata={'help': 'The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}'})