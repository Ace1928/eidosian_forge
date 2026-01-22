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
class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    """Reference type promotion rule from torch.ops.aten.all or torch.ops.aten.any.

    This is a special case where computation dtype is always torch.bool.
    The result dtype is always uint8 if `dtype` kwarg is uint8, otherwise torch.bool.
    """

    def __init__(self, op_name: str):
        super().__init__('aten', op_name, _prims_common.REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL)

    def preview_type_promotion(self, args: tuple, kwargs: dict) -> TypePromotionSnapshot:
        assert len(args) >= 1 and isinstance((arg := args[0]), torch.Tensor), f'Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument'
        computation_dtype = torch.bool
        result_dtype = torch.uint8 if arg.dtype == torch.uint8 else torch.bool
        return TypePromotionSnapshot({0: computation_dtype}, {}, result_dtype)