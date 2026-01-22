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
def af2_idx_to_esm_idx(self, aa, mask):
    if self.af2_to_esm.device != aa.device:
        self.af2_to_esm = self.af2_to_esm.to(aa.device)
    aa = (aa + 1).masked_fill(mask != 1, 0)
    return self.af2_to_esm[aa]