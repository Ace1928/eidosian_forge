import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def torch_stack(tensors, dim=None, axis=None, *, out=None):
    if dim is None and axis is None:
        dim = 0
    if dim is None and axis is not None:
        dim = axis
    if dim < 0:
        dim = tensors[0].dim() + 1 + dim
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return torch.empty(shape, device='meta')