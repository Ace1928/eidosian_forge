from __future__ import annotations
import ast
import inspect
from abc import ABCMeta
from functools import wraps
from pathlib import Path
from jinja2 import Template
from gradio.events import EventListener
from gradio.exceptions import ComponentDefinitionError
from gradio.utils import no_raise_exception
def updateable(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn_args = inspect.getfullargspec(fn).args
        self = args[0]
        initialized_before = hasattr(self, '_constructor_args')
        if not initialized_before:
            self._constructor_args = []
        for i, arg in enumerate(args):
            if i == 0 or i >= len(fn_args):
                continue
            arg_name = fn_args[i]
            kwargs[arg_name] = arg
        self._constructor_args.append(kwargs)
        if in_event_listener() and initialized_before:
            return None
        else:
            return fn(self, **kwargs)
    return wrapper