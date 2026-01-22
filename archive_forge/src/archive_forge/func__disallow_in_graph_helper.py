from typing import TYPE_CHECKING
import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage
def _disallow_in_graph_helper(throw_if_not_allowed):

    def inner(fn):
        if isinstance(fn, (list, tuple)):
            return [disallow_in_graph(x) for x in fn]
        assert callable(fn), 'disallow_in_graph expects a callable'
        if throw_if_not_allowed and (not allowed_functions.is_allowed(fn)):
            raise IncorrectUsage('disallow_in_graph is expected to be used on an already allowed callable (like torch.* ops). Allowed callables means callables that TorchDynamo puts as-is in the extracted graph.')
        allowed_functions._allowed_function_ids.remove(id(fn))
        allowed_functions._disallowed_function_ids.add(id(fn))
        allowed_functions._allowed_user_defined_function_ids.remove(id(fn))
        return fn
    return inner