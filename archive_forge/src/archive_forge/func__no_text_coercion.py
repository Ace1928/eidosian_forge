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
def _no_text_coercion(element: Any, argname: Optional[str]=None, exc_cls: Type[exc.SQLAlchemyError]=exc.ArgumentError, extra: Optional[str]=None, err: Optional[Exception]=None) -> NoReturn:
    raise exc_cls('%(extra)sTextual SQL expression %(expr)r %(argname)sshould be explicitly declared as text(%(expr)r)' % {'expr': util.ellipses_string(element), 'argname': 'for argument %s' % (argname,) if argname else '', 'extra': '%s ' % extra if extra else ''}) from err