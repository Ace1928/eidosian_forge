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
def _extract_declared_columns(self) -> None:
    our_stuff = self.properties
    declared_columns = self.declared_columns
    column_ordering = self.column_ordering
    name_to_prop_key = collections.defaultdict(set)
    for key, c in list(our_stuff.items()):
        if isinstance(c, _MapsColumns):
            mp_to_assign = c.mapper_property_to_assign
            if mp_to_assign:
                our_stuff[key] = mp_to_assign
            else:
                del our_stuff[key]
            for col, sort_order in c.columns_to_assign:
                if not isinstance(c, CompositeProperty):
                    name_to_prop_key[col.name].add(key)
                declared_columns.add(col)
                column_ordering[col] = sort_order
                if mp_to_assign is None and key != col.key:
                    our_stuff[key] = col
        elif isinstance(c, Column):
            assert c.name is not None
            name_to_prop_key[c.name].add(key)
            declared_columns.add(c)
            if key == c.key:
                del our_stuff[key]
    for name, keys in name_to_prop_key.items():
        if len(keys) > 1:
            util.warn('On class %r, Column object %r named directly multiple times, only one will be used: %s. Consider using orm.synonym instead' % (self.classname, name, ', '.join(sorted(keys))))