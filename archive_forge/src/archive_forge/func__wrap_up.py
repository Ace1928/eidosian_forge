import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
    if self.linear_g is not None:
        g = self.sigmoid(self.linear_g(q_x))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))
        o = o * g
    o = flatten_final_dims(o, 2)
    o = self.linear_o(o)
    return o