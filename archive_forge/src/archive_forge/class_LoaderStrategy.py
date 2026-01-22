from __future__ import annotations
import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict
class LoaderStrategy:
    """Describe the loading behavior of a StrategizedProperty object.

    The ``LoaderStrategy`` interacts with the querying process in three
    ways:

    * it controls the configuration of the ``InstrumentedAttribute``
      placed on a class to handle the behavior of the attribute.  this
      may involve setting up class-level callable functions to fire
      off a select operation when the attribute is first accessed
      (i.e. a lazy load)

    * it processes the ``QueryContext`` at statement construction time,
      where it can modify the SQL statement that is being produced.
      For example, simple column attributes will add their represented
      column to the list of selected columns, a joined eager loader
      may establish join clauses to add to the statement.

    * It produces "row processor" functions at result fetching time.
      These "row processor" functions populate a particular attribute
      on a particular mapped instance.

    """
    __slots__ = ('parent_property', 'is_class_level', 'parent', 'key', 'strategy_key', 'strategy_opts')
    _strategy_keys: ClassVar[List[_StrategyKey]]

    def __init__(self, parent: MapperProperty[Any], strategy_key: _StrategyKey):
        self.parent_property = parent
        self.is_class_level = False
        self.parent = self.parent_property.parent
        self.key = self.parent_property.key
        self.strategy_key = strategy_key
        self.strategy_opts = dict(strategy_key)

    def init_class_attribute(self, mapper: Mapper[Any]) -> None:
        pass

    def setup_query(self, compile_state: ORMCompileState, query_entity: _MapperEntity, path: AbstractEntityRegistry, loadopt: Optional[_LoadElement], adapter: Optional[ORMAdapter], **kwargs: Any) -> None:
        """Establish column and other state for a given QueryContext.

        This method fulfills the contract specified by MapperProperty.setup().

        StrategizedProperty delegates its setup() method
        directly to this method.

        """

    def create_row_processor(self, context: ORMCompileState, query_entity: _MapperEntity, path: AbstractEntityRegistry, loadopt: Optional[_LoadElement], mapper: Mapper[Any], result: Result[Any], adapter: Optional[ORMAdapter], populators: _PopulatorDict) -> None:
        """Establish row processing functions for a given QueryContext.

        This method fulfills the contract specified by
        MapperProperty.create_row_processor().

        StrategizedProperty delegates its create_row_processor() method
        directly to this method.

        """

    def __str__(self) -> str:
        return str(self.parent_property)