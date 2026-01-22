import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
@property
def __signature__(self):
    if self._sig is None:
        sig = inspect.signature(self.orig_func)
        if not any((p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())):
            sig = sig.replace(parameters=[*sig.parameters.values(), inspect.Parameter('backend', inspect.Parameter.KEYWORD_ONLY, default=None), inspect.Parameter('backend_kwargs', inspect.Parameter.VAR_KEYWORD)])
        else:
            *parameters, var_keyword = sig.parameters.values()
            sig = sig.replace(parameters=[*parameters, inspect.Parameter('backend', inspect.Parameter.KEYWORD_ONLY, default=None), var_keyword])
        self._sig = sig
    return self._sig