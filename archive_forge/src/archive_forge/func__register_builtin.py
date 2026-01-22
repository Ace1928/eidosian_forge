import cmath
import math
import warnings
from collections import OrderedDict
from typing import Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple
def _register_builtin(fn, op):
    _get_builtin_table()[id(fn)] = op