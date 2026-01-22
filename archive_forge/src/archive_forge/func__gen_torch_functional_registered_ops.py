import cmath
import math
import warnings
from collections import OrderedDict
from typing import Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple
def _gen_torch_functional_registered_ops():
    ops = ['stft', 'istft', 'lu', 'cdist', 'norm', 'unique', 'unique_consecutive', 'tensordot']
    return {getattr(torch.functional, name) for name in ops}