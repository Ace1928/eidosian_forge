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
class _MappedAnnotationBase(Generic[_T_co], TypingOnly):
    """common class for Mapped and similar ORM container classes.

    these are classes that can appear on the left side of an ORM declarative
    mapping, containing a mapped class or in some cases a collection
    surrounding a mapped class.

    """
    __slots__ = ()