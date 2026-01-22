from __future__ import annotations
import functools
import math
from typing import Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, KeepDims, normalizer
class LinAlgError(Exception):
    pass