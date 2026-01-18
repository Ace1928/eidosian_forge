import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
def torch_abs_override(input, *, out=None):
    assert out is None, 'Dont support in-place abs for MetaTensor analysis'
    return input