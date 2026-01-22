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
@classmethod
def _parse_torch_refs(cls, ref_module: ModuleType) -> Set[ElementwiseTypePromotionRule]:
    logger.info('Processing module: %s', ref_module.__name__)
    rule_set = set()
    for name in ref_module.__all__:
        decorated_op = getattr(ref_module, name)
        rule = cls._parse_type_promotion_rule_from_refs_op(decorated_op)
        if rule is not None and rule.is_valid():
            rule_set.add(rule)
    return rule_set