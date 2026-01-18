import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
def to_key(parameters):
    return tuple((parameters[k] for k in sorted(parameters)))