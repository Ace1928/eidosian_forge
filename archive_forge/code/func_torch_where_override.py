import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
def torch_where_override(condition, x, y):
    return condition.to(device='meta') + x.to(device='meta') + y.to(device='meta')