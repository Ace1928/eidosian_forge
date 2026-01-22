from __future__ import annotations
import collections.abc as collections_abc
import numbers
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import operators
from . import roles
from . import visitors
from ._typing import is_from_clause
from .base import ExecutableOption
from .base import Options
from .cache_key import HasCacheKey
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
def _deep_is_literal(element):
    """Return whether or not the element is a "literal" in the context
    of a SQL expression construct.

    does a deeper more esoteric check than _is_literal.   is used
    for lambda elements that have to distinguish values that would
    be bound vs. not without any context.

    """
    if isinstance(element, collections_abc.Sequence) and (not isinstance(element, str)):
        for elem in element:
            if not _deep_is_literal(elem):
                return False
        else:
            return True
    return not isinstance(element, (Visitable, schema.SchemaEventTarget, HasCacheKey, Options, util.langhelpers.symbol)) and (not hasattr(element, '__clause_element__')) and (not isinstance(element, type) or not issubclass(element, HasCacheKey))