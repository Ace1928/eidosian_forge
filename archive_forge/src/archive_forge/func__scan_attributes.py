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
def _scan_attributes(self) -> None:
    cls = self.cls
    cls_as_Decl = cast('_DeclMappedClassProtocol[Any]', cls)
    clsdict_view = self.clsdict_view
    collected_attributes = self.collected_attributes
    column_copies = self.column_copies
    _include_dunders = self._include_dunders
    mapper_args_fn = None
    table_args = inherited_table_args = None
    tablename = None
    fixed_table = '__table__' in clsdict_view
    attribute_is_overridden = self._cls_attr_override_checker(self.cls)
    bases = []
    for base in cls.__mro__:
        class_mapped = base is not cls and _is_supercls_for_inherits(base)
        local_attributes_for_class = self._cls_attr_resolver(base)
        if not class_mapped and base is not cls:
            locally_collected_columns = self._produce_column_copies(local_attributes_for_class, attribute_is_overridden, fixed_table, base)
        else:
            locally_collected_columns = {}
        bases.append((base, class_mapped, local_attributes_for_class, locally_collected_columns))
    for base, class_mapped, local_attributes_for_class, locally_collected_columns in bases:
        collected_attributes.update(locally_collected_columns)
        for name, obj, annotation, is_dataclass_field in local_attributes_for_class():
            if name in _include_dunders:
                if name == '__mapper_args__':
                    check_decl = _check_declared_props_nocascade(obj, name, cls)
                    if not mapper_args_fn and (not class_mapped or check_decl):

                        def _mapper_args_fn() -> Dict[str, Any]:
                            return dict(cls_as_Decl.__mapper_args__)
                        mapper_args_fn = _mapper_args_fn
                elif name == '__tablename__':
                    check_decl = _check_declared_props_nocascade(obj, name, cls)
                    if not tablename and (not class_mapped or check_decl):
                        tablename = cls_as_Decl.__tablename__
                elif name == '__table_args__':
                    check_decl = _check_declared_props_nocascade(obj, name, cls)
                    if not table_args and (not class_mapped or check_decl):
                        table_args = cls_as_Decl.__table_args__
                        if not isinstance(table_args, (tuple, dict, type(None))):
                            raise exc.ArgumentError('__table_args__ value must be a tuple, dict, or None')
                        if base is not cls:
                            inherited_table_args = True
                else:
                    continue
            elif class_mapped:
                if _is_declarative_props(obj) and (not obj._quiet):
                    util.warn("Regular (i.e. not __special__) attribute '%s.%s' uses @declared_attr, but owning class %s is mapped - not applying to subclass %s." % (base.__name__, name, base, cls))
                continue
            elif base is not cls:
                if isinstance(obj, (Column, MappedColumn)):
                    continue
                elif isinstance(obj, MapperProperty):
                    raise exc.InvalidRequestError('Mapper properties (i.e. deferred,column_property(), relationship(), etc.) must be declared as @declared_attr callables on declarative mixin classes.  For dataclass field() objects, use a lambda:')
                elif _is_declarative_props(obj):
                    assert obj is not None
                    if obj._cascading:
                        if name in clsdict_view:
                            util.warn("Attribute '%s' on class %s cannot be processed due to @declared_attr.cascading; skipping" % (name, cls))
                        collected_attributes[name] = column_copies[obj] = ret = obj.__get__(obj, cls)
                        setattr(cls, name, ret)
                    else:
                        if is_dataclass_field:
                            ret = getattr(cls, name, None)
                            if not isinstance(ret, InspectionAttr):
                                ret = obj.fget()
                        else:
                            ret = getattr(cls, name)
                        if isinstance(ret, InspectionAttr) and attr_is_internal_proxy(ret) and (not isinstance(ret.original_property, MapperProperty)):
                            ret = ret.descriptor
                        collected_attributes[name] = column_copies[obj] = ret
                    if isinstance(ret, (Column, MapperProperty)) and ret.doc is None:
                        ret.doc = obj.__doc__
                    self._collect_annotation(name, obj._collect_return_annotation(), base, True, obj)
                elif _is_mapped_annotation(annotation, cls, base):
                    if not fixed_table:
                        assert name in collected_attributes or attribute_is_overridden(name, None)
                    continue
                else:
                    self._warn_for_decl_attributes(base, name, obj)
            elif is_dataclass_field and (name not in clsdict_view or clsdict_view[name] is not obj):
                assert not attribute_is_overridden(name, obj)
                if _is_declarative_props(obj):
                    obj = obj.fget()
                collected_attributes[name] = obj
                self._collect_annotation(name, annotation, base, False, obj)
            else:
                collected_annotation = self._collect_annotation(name, annotation, base, None, obj)
                is_mapped = collected_annotation is not None and collected_annotation.mapped_container is not None
                generated_obj = collected_annotation.attr_value if collected_annotation is not None else obj
                if obj is None and (not fixed_table) and is_mapped:
                    collected_attributes[name] = generated_obj if generated_obj is not None else MappedColumn()
                elif name in clsdict_view:
                    collected_attributes[name] = obj
    if inherited_table_args and (not tablename):
        table_args = None
    self.table_args = table_args
    self.tablename = tablename
    self.mapper_args_fn = mapper_args_fn