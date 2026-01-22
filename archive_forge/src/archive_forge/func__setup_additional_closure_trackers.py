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
def _setup_additional_closure_trackers(self, fn, lambda_element, opts):
    analyzed_function = AnalyzedFunction(self, lambda_element, None, fn)
    closure_trackers = self.closure_trackers
    for pywrapper in analyzed_function.closure_pywrappers:
        if not pywrapper._sa__has_param:
            closure_trackers.append(self._cache_key_getter_tracked_literal(fn, pywrapper))