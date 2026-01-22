import copy
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
from collections import namedtuple
from typing import Callable, Dict, List, Union
from .backend_config import (
from ..fuser_method_mappings import (
def _get_share_qprams_op_backend_config(op):
    return BackendPatternConfig(op).set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT).set_dtype_configs(dtype_configs)