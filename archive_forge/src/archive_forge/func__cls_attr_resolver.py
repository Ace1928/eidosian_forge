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
def _cls_attr_resolver(self, cls: Type[Any]) -> Callable[[], Iterable[Tuple[str, Any, Any, bool]]]:
    """produce a function to iterate the "attributes" of a class
        which we want to consider for mapping, adjusting for SQLAlchemy fields
        embedded in dataclass fields.

        """
    cls_annotations = util.get_annotations(cls)
    cls_vars = vars(cls)
    _include_dunders = self._include_dunders
    _match_exclude_dunders = self._match_exclude_dunders
    names = [n for n in util.merge_lists_w_ordering(list(cls_vars), list(cls_annotations)) if not _match_exclude_dunders.match(n) or n in _include_dunders]
    if self.allow_dataclass_fields:
        sa_dataclass_metadata_key: Optional[str] = _get_immediate_cls_attr(cls, '__sa_dataclass_metadata_key__')
    else:
        sa_dataclass_metadata_key = None
    if not sa_dataclass_metadata_key:

        def local_attributes_for_class() -> Iterable[Tuple[str, Any, Any, bool]]:
            return ((name, cls_vars.get(name), cls_annotations.get(name), False) for name in names)
    else:
        dataclass_fields = {field.name: field for field in util.local_dataclass_fields(cls)}
        fixed_sa_dataclass_metadata_key = sa_dataclass_metadata_key

        def local_attributes_for_class() -> Iterable[Tuple[str, Any, Any, bool]]:
            for name in names:
                field = dataclass_fields.get(name, None)
                if field and sa_dataclass_metadata_key in field.metadata:
                    yield (field.name, _as_dc_declaredattr(field.metadata, fixed_sa_dataclass_metadata_key), cls_annotations.get(field.name), True)
                else:
                    yield (name, cls_vars.get(name), cls_annotations.get(name), False)
    return local_attributes_for_class