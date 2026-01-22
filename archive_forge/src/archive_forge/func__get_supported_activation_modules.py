from itertools import chain
from operator import getitem
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union
from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param
from .match_utils import apply_match, MatchAllNode
from .prune_functions import (
def _get_supported_activation_modules():
    SUPPORTED_ACTIVATION_MODULES = {nn.ReLU, nn.RReLU, nn.Hardtanh, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid, nn.Tanh, nn.SiLU, nn.Mish, nn.Hardswish, nn.ELU, nn.CELU, nn.SELU, nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid, nn.Softplus, nn.PReLU, nn.Softsign, nn.Tanhshrink, nn.GELU}
    return SUPPORTED_ACTIVATION_MODULES