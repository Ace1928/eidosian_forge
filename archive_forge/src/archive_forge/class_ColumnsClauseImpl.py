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
class ColumnsClauseImpl(_SelectIsNotFrom, _CoerceLiterals, RoleImpl):
    __slots__ = ()
    _coerce_consts = True
    _coerce_numerics = True
    _coerce_star = True
    _guess_straight_column = re.compile('^\\w\\S*$', re.I)

    def _raise_for_expected(self, element, argname=None, resolved=None, advice=None, **kw):
        if not advice and isinstance(element, list):
            advice = f'Did you mean to say select({', '.join((repr(e) for e in element))})?'
        return super()._raise_for_expected(element, argname=argname, resolved=resolved, advice=advice, **kw)

    def _text_coercion(self, element, argname=None):
        element = str(element)
        guess_is_literal = not self._guess_straight_column.match(element)
        raise exc.ArgumentError('Textual column expression %(column)r %(argname)sshould be explicitly declared with text(%(column)r), or use %(literal_column)s(%(column)r) for more specificity' % {'column': util.ellipses_string(element), 'argname': 'for argument %s' % (argname,) if argname else '', 'literal_column': 'literal_column' if guess_is_literal else 'column'})