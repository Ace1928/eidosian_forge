import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
class elementwise_type_promotion_wrapper:
    """
    Adds elementwise type promotion to a Python reference implementation.

    Takes two kwargs, type_promoting_args and type_promotion_kind.

    type_promoting_args must be a string Sequence specifiying the argument names of all
    arguments that participate in type promotion (and should be type promoted). If the
    arg specifies a Sequence-type then every element of the Sequence will participate in
    type promotion.

    type_promotion_kind must be one of the kinds specified by ELEMENTWISE_TYPE_PROMOTION_KIND.
    See its documentation for details.

    The return_dtype will be coerced to the wrapped function's dtype arg if it is available and
    not None.

    Other type promotion behavior, like validating the Python type of scalar arguments, must
    be handled separately.
    """

    def __init__(self, *, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND, type_promoting_args: Optional[Sequence[str]]=None):
        self.type_promoting_arg_names = type_promoting_args
        self.type_promotion_kind = type_promotion_kind

    def __call__(self, fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @wraps(fn)
        def _fn(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            type_promoting_args = tuple((bound.arguments[x] for x in self.type_promoting_arg_names if x in bound.arguments.keys()))
            flattened_type_promoting_args = pytree.arg_tree_leaves(*type_promoting_args)
            compute_dtype, result_dtype = utils.elementwise_dtypes(*flattened_type_promoting_args, type_promotion_kind=self.type_promotion_kind)
            promoted_args = {x: _maybe_convert_to_dtype(bound.arguments[x], compute_dtype) for x in self.type_promoting_arg_names if x in bound.arguments.keys()}
            bound.arguments.update(promoted_args)
            result = fn(**bound.arguments)
            if 'dtype' in bound.arguments:
                maybe_dtype = bound.arguments['dtype']
                if maybe_dtype:
                    result_dtype = maybe_dtype
            if isinstance(result, TensorLike):
                return _maybe_convert_to_dtype(result, result_dtype)
            if isinstance(result, Sequence):
                return tuple((_maybe_convert_to_dtype(x, result_dtype) for x in result))
            raise AssertionError(f'Unhandled result type: {type(result)}')
        _fn.__signature__ = sig
        return _fn