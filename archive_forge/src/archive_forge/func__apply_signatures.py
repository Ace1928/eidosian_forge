import ast
import copy
import importlib
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from weakref import WeakKeyDictionary
import param
from bokeh.core.has_props import _default_resolver
from bokeh.document import Document
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from pyviz_comms import (
from .io.logging import panel_log_handler
from .io.state import state
from .util import param_watchers
def _apply_signatures(self):
    from inspect import Parameter, Signature
    from .viewable import Viewable
    descendants = param.concrete_descendents(Viewable)
    for cls in reversed(list(descendants.values())):
        if cls.__doc__ is None:
            pass
        elif cls.__doc__.startswith('params'):
            prefix = cls.__doc__.split('\n')[0]
            cls.__doc__ = cls.__doc__.replace(prefix, '')
        sig = inspect.signature(cls.__init__)
        sig_params = list(sig.parameters.values())
        if not sig_params or sig_params[-1] != Parameter('params', Parameter.VAR_KEYWORD):
            continue
        parameters = sig_params[:-1]
        processed_kws, keyword_groups = (set(), [])
        for scls in reversed(cls.mro()):
            keyword_group = []
            for k, v in sorted(scls.__dict__.items()):
                if isinstance(v, param.Parameter) and k not in processed_kws and (not v.readonly):
                    keyword_group.append(k)
                    processed_kws.add(k)
            keyword_groups.append(keyword_group)
        parameters += [Parameter(name, Parameter.KEYWORD_ONLY) for kws in reversed(keyword_groups) for name in kws if name not in sig.parameters]
        kwarg_name = '_kwargs' if 'kwargs' in processed_kws else 'kwargs'
        parameters.append(Parameter(kwarg_name, Parameter.VAR_KEYWORD))
        cls.__init__.__signature__ = Signature(parameters, return_annotation=sig.return_annotation)