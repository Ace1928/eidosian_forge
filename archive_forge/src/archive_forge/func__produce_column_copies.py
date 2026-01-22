from __future__ import annotations
import collections
import dataclasses
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import exc as orm_exc
from . import instrumentation
from . import mapperlib
from ._typing import _O
from ._typing import attr_is_internal_proxy
from .attributes import InstrumentedAttribute
from .attributes import QueryableAttribute
from .base import _is_mapped_class
from .base import InspectionAttr
from .descriptor_props import CompositeProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DCAttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MappedAttribute
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .mapper import Mapper
from .properties import ColumnProperty
from .properties import MappedColumn
from .util import _extract_mapped_subtype
from .util import _is_mapped_annotation
from .util import class_mapper
from .util import de_stringify_annotation
from .. import event
from .. import exc
from .. import util
from ..sql import expression
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import Table
from ..util import topological
from ..util.typing import _AnnotationScanType
from ..util.typing import is_fwd_ref
from ..util.typing import is_literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
from ..util.typing import typing_get_args
def _produce_column_copies(self, attributes_for_class: Callable[[], Iterable[Tuple[str, Any, Any, bool]]], attribute_is_overridden: Callable[[str, Any], bool], fixed_table: bool, originating_class: Type[Any]) -> Dict[str, Union[Column[Any], MappedColumn[Any]]]:
    cls = self.cls
    dict_ = self.clsdict_view
    locally_collected_attributes = {}
    column_copies = self.column_copies
    for name, obj, annotation, is_dataclass in attributes_for_class():
        if not fixed_table and obj is None and _is_mapped_annotation(annotation, cls, originating_class):
            if attribute_is_overridden(name, obj):
                continue
            collected_annotation = self._collect_annotation(name, annotation, originating_class, True, obj)
            obj = collected_annotation.attr_value if collected_annotation is not None else obj
            if obj is None:
                obj = MappedColumn()
            locally_collected_attributes[name] = obj
            setattr(cls, name, obj)
        elif isinstance(obj, (Column, MappedColumn)):
            if attribute_is_overridden(name, obj):
                continue
            collected_annotation = self._collect_annotation(name, annotation, originating_class, True, obj)
            obj = collected_annotation.attr_value if collected_annotation is not None else obj
            if name not in dict_ and (not ('__table__' in dict_ and (getattr(obj, 'name', None) or name) in dict_['__table__'].c)):
                if obj.foreign_keys:
                    for fk in obj.foreign_keys:
                        if fk._table_column is not None and fk._table_column.table is None:
                            raise exc.InvalidRequestError('Columns with foreign keys to non-table-bound columns must be declared as @declared_attr callables on declarative mixin classes.  For dataclass field() objects, use a lambda:.')
                column_copies[obj] = copy_ = obj._copy()
                locally_collected_attributes[name] = copy_
                setattr(cls, name, copy_)
    return locally_collected_attributes