from __future__ import annotations
import tokenize
from typing import TYPE_CHECKING
import warnings
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import is_extension_array_dtype
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.parsing import tokenize_string
from pandas.core.computation.scope import ensure_scope
from pandas.core.generic import NDFrame
from pandas.io.formats.printing import pprint_thing
def _check_resolvers(resolvers):
    if resolvers is not None:
        for resolver in resolvers:
            if not hasattr(resolver, '__getitem__'):
                name = type(resolver).__name__
                raise TypeError(f"Resolver of type '{name}' does not implement the __getitem__ method")