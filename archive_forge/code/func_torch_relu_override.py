import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
def torch_relu_override(x):
    return x