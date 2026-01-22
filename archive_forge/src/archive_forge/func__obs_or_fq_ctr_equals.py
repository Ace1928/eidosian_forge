from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _obs_or_fq_ctr_equals(obs_or_fq1: _ObserverOrFakeQuantizeConstructor, obs_or_fq2: _ObserverOrFakeQuantizeConstructor):
    if isinstance(obs_or_fq1, _PartialWrapper) and isinstance(obs_or_fq2, _PartialWrapper):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    return obs_or_fq1 == obs_or_fq2