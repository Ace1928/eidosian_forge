import cmath
import math
import warnings
from collections import OrderedDict
from typing import Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple
def _is_special_functional_bound_op(fn):
    return fn in _functional_registered_ops