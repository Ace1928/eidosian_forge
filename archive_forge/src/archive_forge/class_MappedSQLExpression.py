from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import strategy_options
from .base import _DeclarativeMapped
from .base import class_mapper
from .descriptor_props import CompositeProperty
from .descriptor_props import ConcreteInheritedProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .interfaces import StrategizedProperty
from .relationships import RelationshipProperty
from .util import de_stringify_annotation
from .util import de_stringify_union_elements
from .. import exc as sa_exc
from .. import ForeignKey
from .. import log
from .. import util
from ..sql import coercions
from ..sql import roles
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import SchemaConst
from ..sql.type_api import TypeEngine
from ..util.typing import de_optionalize_union_types
from ..util.typing import is_fwd_ref
from ..util.typing import is_optional_union
from ..util.typing import is_pep593
from ..util.typing import is_pep695
from ..util.typing import is_union
from ..util.typing import Self
from ..util.typing import typing_get_args
class MappedSQLExpression(ColumnProperty[_T], _DeclarativeMapped[_T]):
    """Declarative front-end for the :class:`.ColumnProperty` class.

    Public constructor is the :func:`_orm.column_property` function.

    .. versionchanged:: 2.0 Added :class:`_orm.MappedSQLExpression` as
       a Declarative compatible subclass for :class:`_orm.ColumnProperty`.

    .. seealso::

        :class:`.MappedColumn`

    """
    inherit_cache = True
    ':meta private:'