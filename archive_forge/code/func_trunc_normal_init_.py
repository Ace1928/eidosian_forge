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
def trunc_normal_init_(weights, scale=1.0, fan='fan_in'):
    shape = weights.shape
    scale = scale / max(1, shape[1])
    if not is_scipy_available():
        logger.warning('This init requires scipy, but scipy was not found, default to an approximation that might not be equivalent.')
        std = math.sqrt(scale)
        torch.nn.init.normal_(weights, std=std).clamp(min=0.0, max=2.0 * std)
    else:
        from scipy.stats import truncnorm
        std = math.sqrt(scale) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
        samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=weights.numel())
        samples = np.reshape(samples, shape)
        weights.copy_(torch.tensor(samples, device=weights.device))