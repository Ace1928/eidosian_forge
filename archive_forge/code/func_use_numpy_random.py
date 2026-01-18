from __future__ import annotations
import functools
from math import sqrt
from typing import Optional
import torch
from . import _dtypes_impl, _util
from ._normalizations import array_or_scalar, ArrayLike, normalizer
def use_numpy_random():
    import torch._dynamo.config as config
    return config.use_numpy_random_stream