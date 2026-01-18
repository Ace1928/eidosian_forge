import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longt5 import LongT5Config
reshape