from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def _silu_backward(dy, x):
    sigm = 1 / (1 + torch.exp(-x.float()))
    return (dy.float() * sigm * (1 + x.float() * (1 - sigm))).to(x.dtype)