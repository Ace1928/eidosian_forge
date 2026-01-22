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
class StatementImpl(_CoerceLiterals, RoleImpl):
    __slots__ = ()

    def _post_coercion(self, resolved, original_element, argname=None, **kw):
        if resolved is not original_element and (not isinstance(original_element, str)):
            try:
                original_element._execute_on_connection
            except AttributeError:
                util.warn_deprecated('Object %r should not be used directly in a SQL statement context, such as passing to methods such as session.execute().  This usage will be disallowed in a future release.  Please use Core select() / update() / delete() etc. with Session.execute() and other statement execution methods.' % original_element, '1.4')
        return resolved

    def _implicit_coercions(self, element: Any, resolved: Any, argname: Optional[str]=None, **kw: Any) -> Any:
        if resolved._is_lambda_element:
            return resolved
        else:
            return super()._implicit_coercions(element, resolved, argname=argname, **kw)