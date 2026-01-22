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
class ElementwiseTypePromotionRuleSetGenerator:
    """Hackly distilling info from reference ops decorated with elementwise type promotion rule.

    The goal is to retrieve the decorator

    ```python
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
    ```

    from the reference ops. It provides info as for which arguments are promoted
    and what kind of promotion is applied.
    """

    @classmethod
    def generate_from_torch_refs(cls) -> Set[ElementwiseTypePromotionRule]:
        """Parse type promotion rules from reference ops under torch._C._refs."""
        rule_set = set()
        rule_set.update(cls._parse_torch_refs(_refs))
        rule_set.update(cls._parse_torch_refs(_nn_refs))
        rule_set.update(cls._parse_torch_refs(_linalg_refs))
        rule_set.update(cls._parse_torch_refs(_special_refs))
        rule_set.update(cls._parse_torch_refs(_functional_refs))
        return rule_set

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

    @classmethod
    def _parse_type_promotion_rule_from_refs_op(cls, decorated_op: Callable) -> Optional[ElementwiseTypePromotionRule]:
        """Retrieve and parse type promotion decorator from op under torch._refs."""
        fn = decorated_op
        type_promo_wrapper = None
        while (fn_closure_vars := _try_getclosurevars(fn)):
            if 'fn' not in fn_closure_vars.nonlocals:
                break
            if 'self' in fn_closure_vars.nonlocals and isinstance(fn_closure_vars.nonlocals['self'], _prims_common_wrappers.elementwise_type_promotion_wrapper):
                type_promo_wrapper = fn_closure_vars.nonlocals['self']
                break
            fn = fn_closure_vars.nonlocals['fn']
        if type_promo_wrapper is not None:
            signature = inspect.signature(decorated_op)
            pos = 0
            promote_args_positions = []
            promote_kwargs_names = []
            if type_promo_wrapper.type_promoting_arg_names is not None:
                for name, param in signature.parameters.items():
                    if name in type_promo_wrapper.type_promoting_arg_names:
                        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY):
                            promote_args_positions.append(pos)
                        elif param.kind == param.KEYWORD_ONLY:
                            promote_kwargs_names.append(name)
                    pos += 1
            return ElementwiseTypePromotionRule('aten', decorated_op.__name__, promote_args_positions=promote_args_positions, promote_kwargs_names=promote_kwargs_names, promotion_kind=type_promo_wrapper.type_promotion_kind)
        logger.warning('Cannot find type promotion rule for: %s.%s', decorated_op.__module__, decorated_op.__name__)
        return None