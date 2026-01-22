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
class RelationshipDirection(Enum):
    """enumeration which indicates the 'direction' of a
    :class:`_orm.RelationshipProperty`.

    :class:`.RelationshipDirection` is accessible from the
    :attr:`_orm.Relationship.direction` attribute of
    :class:`_orm.RelationshipProperty`.

    """
    ONETOMANY = 1
    'Indicates the one-to-many direction for a :func:`_orm.relationship`.\n\n    This symbol is typically used by the internals but may be exposed within\n    certain API features.\n\n    '
    MANYTOONE = 2
    'Indicates the many-to-one direction for a :func:`_orm.relationship`.\n\n    This symbol is typically used by the internals but may be exposed within\n    certain API features.\n\n    '
    MANYTOMANY = 3
    'Indicates the many-to-many direction for a :func:`_orm.relationship`.\n\n    This symbol is typically used by the internals but may be exposed within\n    certain API features.\n\n    '