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
def _setup_inheriting_mapper(self, mapper_kw: _MapperKwArgs) -> None:
    cls = self.cls
    inherits = mapper_kw.get('inherits', None)
    if inherits is None:
        inherits_search = []
        for base_ in cls.__bases__:
            c = _resolve_for_abstract_or_classical(base_)
            if c is None:
                continue
            if _is_supercls_for_inherits(c) and c not in inherits_search:
                inherits_search.append(c)
        if inherits_search:
            if len(inherits_search) > 1:
                raise exc.InvalidRequestError('Class %s has multiple mapped bases: %r' % (cls, inherits_search))
            inherits = inherits_search[0]
    elif isinstance(inherits, Mapper):
        inherits = inherits.class_
    self.inherits = inherits
    clsdict_view = self.clsdict_view
    if '__table__' not in clsdict_view and self.tablename is None:
        self.single = True