import math
import warnings
from collections import OrderedDict
import torch
from packaging import version
from torch import Tensor, nn
from .utils import logging
class SiLUActivation(nn.SiLU):

    def __init__(self, *args, **kwargs):
        warnings.warn('The SiLUActivation class has been deprecated and will be removed in v4.39. Please use nn.SiLU instead.')
        super().__init__(*args, **kwargs)