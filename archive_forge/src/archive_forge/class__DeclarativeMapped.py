from __future__ import annotations
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import no_type_check
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc
from ._typing import insp_is_mapper
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import roles
from ..sql.elements import SQLColumnExpression
from ..sql.elements import SQLCoreOperations
from ..util import FastIntFlag
from ..util.langhelpers import TypingOnly
from ..util.typing import Literal
class _DeclarativeMapped(Mapped[_T_co], _MappedAttribute[_T_co]):
    """Mixin for :class:`.MapperProperty` subclasses that allows them to
    be compatible with ORM-annotated declarative mappings.

    """
    __slots__ = ()

    def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> Any:
        return NotImplemented
    __sa_operate__ = operate

    def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> Any:
        return NotImplemented