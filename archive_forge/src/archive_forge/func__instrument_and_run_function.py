from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def _instrument_and_run_function(self, lambda_element):
    analyzed_code = self.analyzed_code
    fn = self.fn
    self.closure_pywrappers = closure_pywrappers = []
    build_py_wrappers = analyzed_code.build_py_wrappers
    if not build_py_wrappers:
        self.tracker_instrumented_fn = tracker_instrumented_fn = fn
        self.expr = lambda_element._invoke_user_fn(tracker_instrumented_fn)
    else:
        track_closure_variables = analyzed_code.track_closure_variables
        closure = fn.__closure__
        if closure:
            new_closure = {fv: cell.cell_contents for fv, cell in zip(fn.__code__.co_freevars, closure)}
        else:
            new_closure = {}
        new_globals = fn.__globals__.copy()
        for name, closure_index in build_py_wrappers:
            if closure_index is not None:
                value = closure[closure_index].cell_contents
                new_closure[name] = bind = PyWrapper(fn, name, value, closure_index=closure_index, track_bound_values=self.analyzed_code.track_bound_values)
                if track_closure_variables:
                    closure_pywrappers.append(bind)
            else:
                value = fn.__globals__[name]
                new_globals[name] = bind = PyWrapper(fn, name, value)
        self.tracker_instrumented_fn = tracker_instrumented_fn = self._rewrite_code_obj(fn, [new_closure[name] for name in fn.__code__.co_freevars], new_globals)
        self.expr = lambda_element._invoke_user_fn(tracker_instrumented_fn)