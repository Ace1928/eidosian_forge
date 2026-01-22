from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
def _consolidate_input_dtype(self, computed_dtype: torch.dtype, result_dtype: torch.dtype) -> torch.dtype:
    """
        Although opmath is the right thing to do to retain on-par precision, it inserts
        upcasts everywhere in the graph. This is particularly hard for backend to optimize
        since there is no way to differentiate between inserted upcasts and model code
        casts. Hence we consolidate the input dtype to the result dtype to avoid this.
        """
    if not self._USE_OPMATH and self.promotion_kind == _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return result_dtype
    return computed_dtype