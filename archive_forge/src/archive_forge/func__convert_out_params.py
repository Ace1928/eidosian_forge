import inspect
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, List, Sequence, Union
import torch
import torch.library
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch.utils import _pytree as pytree
import torch._decomp.decompositions
import torch._refs
def _convert_out_params(f):
    out_annotation = f.__annotations__.get('out')
    if not out_annotation:
        return f
    if getattr(out_annotation, '__origin__', None) is tuple:
        sig = inspect.signature(f)
        out_names = sig.return_annotation._fields

        @wraps(f)
        def _fn(*args, **kwargs):
            out_kwargs = tuple((kwargs.pop(o, None) for o in out_names))
            is_none = out_kwargs[0] is None
            assert all(((o is None) == is_none for o in out_kwargs))
            return f(*args, **kwargs, out=None if is_none else out_kwargs)
        out_params = [inspect.Parameter(o, kind=inspect.Parameter.KEYWORD_ONLY, default=None, annotation=t) for o, t in zip(out_names, out_annotation.__args__)]
        params = chain((v for k, v in sig.parameters.items() if k != 'out'), out_params)
        _fn.__signature__ = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != 'out'}
        for o in out_params:
            _fn.__annotations__[o.name] = o.annotation
        _fn._torch_decompositions_out_wrapper = f._torch_decompositions_out_wrapper
        return _fn
    custom_out_param_name = f.__annotations__.pop(CustomOutParamAnnotation, None)
    if custom_out_param_name:

        @wraps(f)
        def _fn(*args, **kwargs):
            out_kwarg = kwargs.pop(custom_out_param_name, None)
            return f(*args, **kwargs, out=out_kwarg)
        out_param = inspect.Parameter(custom_out_param_name, kind=inspect.Parameter.KEYWORD_ONLY, default=None, annotation=out_annotation)
        sig = inspect.signature(f)
        params = chain((v for k, v in sig.parameters.items() if k != 'out'), (out_param,))
        _fn.__signature__ = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != 'out'}
        _fn.__annotations__[out_param.name] = out_param.annotation
        return _fn
    return f