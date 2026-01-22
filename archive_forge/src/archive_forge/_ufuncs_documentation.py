from __future__ import annotations
from typing import Optional
import torch
from . import _binary_ufuncs_impl, _dtypes_impl, _unary_ufuncs_impl, _util
from ._normalizations import (
Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    