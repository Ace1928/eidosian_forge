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
class _ClassScanMapperConfig(_MapperConfig):
    __slots__ = ('registry', 'clsdict_view', 'collected_attributes', 'collected_annotations', 'local_table', 'persist_selectable', 'declared_columns', 'column_ordering', 'column_copies', 'table_args', 'tablename', 'mapper_args', 'mapper_args_fn', 'inherits', 'single', 'allow_dataclass_fields', 'dataclass_setup_arguments', 'is_dataclass_prior_to_mapping', 'allow_unmapped_annotations')
    is_deferred = False
    registry: _RegistryType
    clsdict_view: _ClassDict
    collected_annotations: Dict[str, _CollectedAnnotation]
    collected_attributes: Dict[str, Any]
    local_table: Optional[FromClause]
    persist_selectable: Optional[FromClause]
    declared_columns: util.OrderedSet[Column[Any]]
    column_ordering: Dict[Column[Any], int]
    column_copies: Dict[Union[MappedColumn[Any], Column[Any]], Union[MappedColumn[Any], Column[Any]]]
    tablename: Optional[str]
    mapper_args: Mapping[str, Any]
    table_args: Optional[_TableArgsType]
    mapper_args_fn: Optional[Callable[[], Dict[str, Any]]]
    inherits: Optional[Type[Any]]
    single: bool
    is_dataclass_prior_to_mapping: bool
    allow_unmapped_annotations: bool
    dataclass_setup_arguments: Optional[_DataclassArguments]
    'if the class has SQLAlchemy native dataclass parameters, where\n    we will turn the class into a dataclass within the declarative mapping\n    process.\n\n    '
    allow_dataclass_fields: bool
    'if true, look for dataclass-processed Field objects on the target\n    class as well as superclasses and extract ORM mapping directives from\n    the "metadata" attribute of each Field.\n\n    if False, dataclass fields can still be used, however they won\'t be\n    mapped.\n\n    '

    def __init__(self, registry: _RegistryType, cls_: Type[_O], dict_: _ClassDict, table: Optional[FromClause], mapper_kw: _MapperKwArgs):
        self.clsdict_view = util.immutabledict(dict_) if dict_ else util.EMPTY_DICT
        super().__init__(registry, cls_, mapper_kw)
        self.registry = registry
        self.persist_selectable = None
        self.collected_attributes = {}
        self.collected_annotations = {}
        self.declared_columns = util.OrderedSet()
        self.column_ordering = {}
        self.column_copies = {}
        self.single = False
        self.dataclass_setup_arguments = dca = getattr(self.cls, '_sa_apply_dc_transforms', None)
        self.allow_unmapped_annotations = getattr(self.cls, '__allow_unmapped__', False) or bool(self.dataclass_setup_arguments)
        self.is_dataclass_prior_to_mapping = cld = dataclasses.is_dataclass(cls_)
        sdk = _get_immediate_cls_attr(cls_, '__sa_dataclass_metadata_key__')
        if (not cld or dca) and sdk:
            raise exc.InvalidRequestError("SQLAlchemy mapped dataclasses can't consume mapping information from dataclass.Field() objects if the immediate class is not already a dataclass.")
        self.allow_dataclass_fields = bool(sdk and cld)
        self._setup_declared_events()
        self._scan_attributes()
        self._setup_dataclasses_transforms()
        with mapperlib._CONFIGURE_MUTEX:
            clsregistry.add_class(self.classname, self.cls, registry._class_registry)
            self._setup_inheriting_mapper(mapper_kw)
            self._extract_mappable_attributes()
            self._extract_declared_columns()
            self._setup_table(table)
            self._setup_inheriting_columns(mapper_kw)
            self._early_mapping(mapper_kw)

    def _setup_declared_events(self) -> None:
        if _get_immediate_cls_attr(self.cls, '__declare_last__'):

            @event.listens_for(Mapper, 'after_configured')
            def after_configured() -> None:
                cast('_DeclMappedClassProtocol[Any]', self.cls).__declare_last__()
        if _get_immediate_cls_attr(self.cls, '__declare_first__'):

            @event.listens_for(Mapper, 'before_configured')
            def before_configured() -> None:
                cast('_DeclMappedClassProtocol[Any]', self.cls).__declare_first__()

    def _cls_attr_override_checker(self, cls: Type[_O]) -> Callable[[str, Any], bool]:
        """Produce a function that checks if a class has overridden an
        attribute, taking SQLAlchemy-enabled dataclass fields into account.

        """
        if self.allow_dataclass_fields:
            sa_dataclass_metadata_key = _get_immediate_cls_attr(cls, '__sa_dataclass_metadata_key__')
        else:
            sa_dataclass_metadata_key = None
        if not sa_dataclass_metadata_key:

            def attribute_is_overridden(key: str, obj: Any) -> bool:
                return getattr(cls, key, obj) is not obj
        else:
            all_datacls_fields = {f.name: f.metadata[sa_dataclass_metadata_key] for f in util.dataclass_fields(cls) if sa_dataclass_metadata_key in f.metadata}
            local_datacls_fields = {f.name: f.metadata[sa_dataclass_metadata_key] for f in util.local_dataclass_fields(cls) if sa_dataclass_metadata_key in f.metadata}
            absent = object()

            def attribute_is_overridden(key: str, obj: Any) -> bool:
                if _is_declarative_props(obj):
                    obj = obj.fget
                ret = local_datacls_fields.get(key, absent)
                if _is_declarative_props(ret):
                    ret = ret.fget
                if ret is obj:
                    return False
                elif ret is not absent:
                    return True
                all_field = all_datacls_fields.get(key, absent)
                ret = getattr(cls, key, obj)
                if ret is obj:
                    return False
                if ret is not absent and isinstance(ret, InstrumentedAttribute):
                    return True
                if all_field is obj:
                    return False
                elif all_field is not absent:
                    return True
                return False
        return attribute_is_overridden
    _include_dunders = {'__table__', '__mapper_args__', '__tablename__', '__table_args__'}
    _match_exclude_dunders = re.compile('^(?:_sa_|__)')

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

    def _setup_dataclasses_transforms(self) -> None:
        dataclass_setup_arguments = self.dataclass_setup_arguments
        if not dataclass_setup_arguments:
            return
        if '__dataclass_fields__' in self.cls.__dict__:
            raise exc.InvalidRequestError(f"Class {self.cls} is already a dataclass; ensure that base classes / decorator styles of establishing dataclasses are not being mixed. This can happen if a class that inherits from 'MappedAsDataclass', even indirectly, is been mapped with '@registry.mapped_as_dataclass'")
        warn_for_non_dc_attrs = collections.defaultdict(list)

        def _allow_dataclass_field(key: str, originating_class: Type[Any]) -> bool:
            if originating_class is not self.cls and '__dataclass_fields__' not in originating_class.__dict__:
                warn_for_non_dc_attrs[originating_class].append(key)
            return True
        manager = instrumentation.manager_of_class(self.cls)
        assert manager is not None
        field_list = [_AttributeOptions._get_arguments_for_make_dataclass(key, anno, mapped_container, self.collected_attributes.get(key, _NoArg.NO_ARG)) for key, anno, mapped_container in ((key, mapped_anno if mapped_anno else raw_anno, mapped_container) for key, (raw_anno, mapped_container, mapped_anno, is_dc, attr_value, originating_module, originating_class) in self.collected_annotations.items() if _allow_dataclass_field(key, originating_class) and (key not in self.collected_attributes or not isinstance(self.collected_attributes[key], QueryableAttribute)))]
        if warn_for_non_dc_attrs:
            for originating_class, non_dc_attrs in warn_for_non_dc_attrs.items():
                util.warn_deprecated(f'When transforming {self.cls} to a dataclass, attribute(s) {', '.join((repr(key) for key in non_dc_attrs))} originates from superclass {originating_class}, which is not a dataclass.  This usage is deprecated and will raise an error in SQLAlchemy 2.1.  When declaring SQLAlchemy Declarative Dataclasses, ensure that all mixin classes and other superclasses which include attributes are also a subclass of MappedAsDataclass.', '2.0', code='dcmx')
        annotations = {}
        defaults = {}
        for item in field_list:
            if len(item) == 2:
                name, tp = item
            elif len(item) == 3:
                name, tp, spec = item
                defaults[name] = spec
            else:
                assert False
            annotations[name] = tp
        for k, v in defaults.items():
            setattr(self.cls, k, v)
        self._apply_dataclasses_to_any_class(dataclass_setup_arguments, self.cls, annotations)

    @classmethod
    def _update_annotations_for_non_mapped_class(cls, klass: Type[_O]) -> Mapping[str, _AnnotationScanType]:
        cls_annotations = util.get_annotations(klass)
        new_anno = {}
        for name, annotation in cls_annotations.items():
            if _is_mapped_annotation(annotation, klass, klass):
                extracted = _extract_mapped_subtype(annotation, klass, klass.__module__, name, type(None), required=False, is_dataclass_field=False, expect_mapped=False)
                if extracted:
                    inner, _ = extracted
                    new_anno[name] = inner
            else:
                new_anno[name] = annotation
        return new_anno

    @classmethod
    def _apply_dataclasses_to_any_class(cls, dataclass_setup_arguments: _DataclassArguments, klass: Type[_O], use_annotations: Mapping[str, _AnnotationScanType]) -> None:
        cls._assert_dc_arguments(dataclass_setup_arguments)
        dataclass_callable = dataclass_setup_arguments['dataclass_callable']
        if dataclass_callable is _NoArg.NO_ARG:
            dataclass_callable = dataclasses.dataclass
        restored: Optional[Any]
        if use_annotations:
            restored = getattr(klass, '__annotations__', None)
            klass.__annotations__ = cast('Dict[str, Any]', use_annotations)
        else:
            restored = None
        try:
            dataclass_callable(klass, **{k: v for k, v in dataclass_setup_arguments.items() if v is not _NoArg.NO_ARG and k != 'dataclass_callable'})
        except (TypeError, ValueError) as ex:
            raise exc.InvalidRequestError(f'Python dataclasses error encountered when creating dataclass for {klass.__name__!r}: {ex!r}. Please refer to Python dataclasses documentation for additional information.', code='dcte') from ex
        finally:
            if use_annotations:
                if restored is None:
                    del klass.__annotations__
                else:
                    klass.__annotations__ = restored

    @classmethod
    def _assert_dc_arguments(cls, arguments: _DataclassArguments) -> None:
        allowed = {'init', 'repr', 'order', 'eq', 'unsafe_hash', 'kw_only', 'match_args', 'dataclass_callable'}
        disallowed_args = set(arguments).difference(allowed)
        if disallowed_args:
            msg = ', '.join((f'{arg!r}' for arg in sorted(disallowed_args)))
            raise exc.ArgumentError(f'Dataclass argument(s) {msg} are not accepted')

    def _collect_annotation(self, name: str, raw_annotation: _AnnotationScanType, originating_class: Type[Any], expect_mapped: Optional[bool], attr_value: Any) -> Optional[_CollectedAnnotation]:
        if name in self.collected_annotations:
            return self.collected_annotations[name]
        if raw_annotation is None:
            return None
        is_dataclass = self.is_dataclass_prior_to_mapping
        allow_unmapped = self.allow_unmapped_annotations
        if expect_mapped is None:
            is_dataclass_field = isinstance(attr_value, dataclasses.Field)
            expect_mapped = not is_dataclass_field and (not allow_unmapped) and (attr_value is None or isinstance(attr_value, _MappedAttribute))
        else:
            is_dataclass_field = False
        is_dataclass_field = False
        extracted = _extract_mapped_subtype(raw_annotation, self.cls, originating_class.__module__, name, type(attr_value), required=False, is_dataclass_field=is_dataclass_field, expect_mapped=expect_mapped and (not is_dataclass))
        if extracted is None:
            return None
        extracted_mapped_annotation, mapped_container = extracted
        if attr_value is None and (not is_literal(extracted_mapped_annotation)):
            for elem in typing_get_args(extracted_mapped_annotation):
                if isinstance(elem, str) or is_fwd_ref(elem, check_generic=True):
                    elem = de_stringify_annotation(self.cls, elem, originating_class.__module__, include_generic=True)
                if isinstance(elem, _IntrospectsAnnotations):
                    attr_value = elem.found_in_pep593_annotated()
        self.collected_annotations[name] = ca = _CollectedAnnotation(raw_annotation, mapped_container, extracted_mapped_annotation, is_dataclass, attr_value, originating_class.__module__, originating_class)
        return ca

    def _warn_for_decl_attributes(self, cls: Type[Any], key: str, c: Any) -> None:
        if isinstance(c, expression.ColumnElement):
            util.warn(f"Attribute '{key}' on class {cls} appears to be a non-schema SQLAlchemy expression object; this won't be part of the declarative mapping. To map arbitrary expressions, use ``column_property()`` or a similar function such as ``deferred()``, ``query_expression()`` etc. ")

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

    def _extract_mappable_attributes(self) -> None:
        cls = self.cls
        collected_attributes = self.collected_attributes
        our_stuff = self.properties
        _include_dunders = self._include_dunders
        late_mapped = _get_immediate_cls_attr(cls, '_sa_decl_prepare_nocascade', strict=True)
        allow_unmapped_annotations = self.allow_unmapped_annotations
        expect_annotations_wo_mapped = allow_unmapped_annotations or self.is_dataclass_prior_to_mapping
        look_for_dataclass_things = bool(self.dataclass_setup_arguments)
        for k in list(collected_attributes):
            if k in _include_dunders:
                continue
            value = collected_attributes[k]
            if _is_declarative_props(value):
                if value._cascading:
                    util.warn("Use of @declared_attr.cascading only applies to Declarative 'mixin' and 'abstract' classes.  Currently, this flag is ignored on mapped class %s" % self.cls)
                value = getattr(cls, k)
            elif isinstance(value, QueryableAttribute) and value.class_ is not cls and (value.key != k):
                value = SynonymProperty(value.key)
                setattr(cls, k, value)
            if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (Column, _MappedAttribute)):
                util.warn("Ignoring declarative-like tuple value of attribute '%s': possibly a copy-and-paste error with a comma accidentally placed at the end of the line?" % k)
                continue
            elif look_for_dataclass_things and isinstance(value, dataclasses.Field):
                continue
            elif not isinstance(value, (Column, _DCAttributeOptions)):
                collected_attributes.pop(k)
                self._warn_for_decl_attributes(cls, k, value)
                if not late_mapped:
                    setattr(cls, k, value)
                continue
            elif k in ('metadata',):
                raise exc.InvalidRequestError(f"Attribute name '{k}' is reserved when using the Declarative API.")
            elif isinstance(value, Column):
                _undefer_column_name(k, self.column_copies.get(value, value))
            else:
                if isinstance(value, _IntrospectsAnnotations):
                    annotation, mapped_container, extracted_mapped_annotation, is_dataclass, attr_value, originating_module, originating_class = self.collected_annotations.get(k, (None, None, None, False, None, None, None))
                    if mapped_container is not None or annotation is None or allow_unmapped_annotations:
                        try:
                            value.declarative_scan(self, self.registry, cls, originating_module, k, mapped_container, annotation, extracted_mapped_annotation, is_dataclass)
                        except NameError as ne:
                            raise exc.ArgumentError(f'Could not resolve all types within mapped annotation: "{annotation}".  Ensure all types are written correctly and are imported within the module in use.') from ne
                    else:
                        assert expect_annotations_wo_mapped
                if isinstance(value, _DCAttributeOptions):
                    if value._has_dataclass_arguments and (not look_for_dataclass_things):
                        if isinstance(value, MapperProperty):
                            argnames = ['init', 'default_factory', 'repr', 'default']
                        else:
                            argnames = ['init', 'default_factory', 'repr']
                        args = {a for a in argnames if getattr(value._attribute_options, f'dataclasses_{a}') is not _NoArg.NO_ARG}
                        raise exc.ArgumentError(f"Attribute '{k}' on class {cls} includes dataclasses argument(s): {', '.join(sorted((repr(a) for a in args)))} but class does not specify SQLAlchemy native dataclass configuration.")
                    if not isinstance(value, (MapperProperty, _MapsColumns)):
                        collected_attributes.pop(k)
                        setattr(cls, k, value)
                        continue
            our_stuff[k] = value

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

    def _setup_table(self, table: Optional[FromClause]=None) -> None:
        cls = self.cls
        cls_as_Decl = cast('MappedClassProtocol[Any]', cls)
        tablename = self.tablename
        table_args = self.table_args
        clsdict_view = self.clsdict_view
        declared_columns = self.declared_columns
        column_ordering = self.column_ordering
        manager = attributes.manager_of_class(cls)
        if '__table__' not in clsdict_view and table is None:
            if hasattr(cls, '__table_cls__'):
                table_cls = cast(Type[Table], util.unbound_method_to_callable(cls.__table_cls__))
            else:
                table_cls = Table
            if tablename is not None:
                args: Tuple[Any, ...] = ()
                table_kw: Dict[str, Any] = {}
                if table_args:
                    if isinstance(table_args, dict):
                        table_kw = table_args
                    elif isinstance(table_args, tuple):
                        if isinstance(table_args[-1], dict):
                            args, table_kw = (table_args[0:-1], table_args[-1])
                        else:
                            args = table_args
                autoload_with = clsdict_view.get('__autoload_with__')
                if autoload_with:
                    table_kw['autoload_with'] = autoload_with
                autoload = clsdict_view.get('__autoload__')
                if autoload:
                    table_kw['autoload'] = True
                sorted_columns = sorted(declared_columns, key=lambda c: column_ordering.get(c, 0))
                table = self.set_cls_attribute('__table__', table_cls(tablename, self._metadata_for_cls(manager), *sorted_columns, *args, **table_kw))
        else:
            if table is None:
                table = cls_as_Decl.__table__
            if declared_columns:
                for c in declared_columns:
                    if not table.c.contains_column(c):
                        raise exc.ArgumentError("Can't add additional column %r when specifying __table__" % c.key)
        self.local_table = table

    def _metadata_for_cls(self, manager: ClassManager[Any]) -> MetaData:
        meta: Optional[MetaData] = getattr(self.cls, 'metadata', None)
        if meta is not None:
            return meta
        else:
            return manager.registry.metadata

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

    def _setup_inheriting_columns(self, mapper_kw: _MapperKwArgs) -> None:
        table = self.local_table
        cls = self.cls
        table_args = self.table_args
        declared_columns = self.declared_columns
        if table is None and self.inherits is None and (not _get_immediate_cls_attr(cls, '__no_table__')):
            raise exc.InvalidRequestError('Class %r does not have a __table__ or __tablename__ specified and does not inherit from an existing table-mapped class.' % cls)
        elif self.inherits:
            inherited_mapper_or_config = _declared_mapping_info(self.inherits)
            assert inherited_mapper_or_config is not None
            inherited_table = inherited_mapper_or_config.local_table
            inherited_persist_selectable = inherited_mapper_or_config.persist_selectable
            if table is None:
                if table_args:
                    raise exc.ArgumentError("Can't place __table_args__ on an inherited class with no table.")
                if declared_columns and (not isinstance(inherited_table, Table)):
                    raise exc.ArgumentError(f"Can't declare columns on single-table-inherited subclass {self.cls}; superclass {self.inherits} is not mapped to a Table")
                for col in declared_columns:
                    assert inherited_table is not None
                    if col.name in inherited_table.c:
                        if inherited_table.c[col.name] is col:
                            continue
                        raise exc.ArgumentError(f"Column '{col}' on class {cls.__name__} conflicts with existing column '{inherited_table.c[col.name]}'.  If using Declarative, consider using the use_existing_column parameter of mapped_column() to resolve conflicts.")
                    if col.primary_key:
                        raise exc.ArgumentError("Can't place primary key columns on an inherited class with no table.")
                    if TYPE_CHECKING:
                        assert isinstance(inherited_table, Table)
                    inherited_table.append_column(col)
                    if inherited_persist_selectable is not None and inherited_persist_selectable is not inherited_table:
                        inherited_persist_selectable._refresh_for_new_column(col)

    def _prepare_mapper_arguments(self, mapper_kw: _MapperKwArgs) -> None:
        properties = self.properties
        if self.mapper_args_fn:
            mapper_args = self.mapper_args_fn()
        else:
            mapper_args = {}
        if mapper_kw:
            mapper_args.update(mapper_kw)
        if 'properties' in mapper_args:
            properties = dict(properties)
            properties.update(mapper_args['properties'])
        for k in ('version_id_col', 'polymorphic_on'):
            if k in mapper_args:
                v = mapper_args[k]
                mapper_args[k] = self.column_copies.get(v, v)
        if 'primary_key' in mapper_args:
            mapper_args['primary_key'] = [self.column_copies.get(v, v) for v in util.to_list(mapper_args['primary_key'])]
        if 'inherits' in mapper_args:
            inherits_arg = mapper_args['inherits']
            if isinstance(inherits_arg, Mapper):
                inherits_arg = inherits_arg.class_
            if inherits_arg is not self.inherits:
                raise exc.InvalidRequestError('mapper inherits argument given for non-inheriting class %s' % mapper_args['inherits'])
        if self.inherits:
            mapper_args['inherits'] = self.inherits
        if self.inherits and (not mapper_args.get('concrete', False)):
            inherited_mapper = class_mapper(self.inherits, False)
            inherited_table = inherited_mapper.local_table
            if 'exclude_properties' not in mapper_args:
                mapper_args['exclude_properties'] = exclude_properties = {c.key for c in inherited_table.c if c not in inherited_mapper._columntoproperty}.union(inherited_mapper.exclude_properties or ())
                exclude_properties.difference_update([c.key for c in self.declared_columns])
            for k, col in list(properties.items()):
                if not isinstance(col, expression.ColumnElement):
                    continue
                if k in inherited_mapper._props:
                    p = inherited_mapper._props[k]
                    if isinstance(p, ColumnProperty):
                        properties[k] = [col] + p.columns
        result_mapper_args = mapper_args.copy()
        result_mapper_args['properties'] = properties
        self.mapper_args = result_mapper_args

    def map(self, mapper_kw: _MapperKwArgs=util.EMPTY_DICT) -> Mapper[Any]:
        self._prepare_mapper_arguments(mapper_kw)
        if hasattr(self.cls, '__mapper_cls__'):
            mapper_cls = cast('Type[Mapper[Any]]', util.unbound_method_to_callable(self.cls.__mapper_cls__))
        else:
            mapper_cls = Mapper
        return self.set_cls_attribute('__mapper__', mapper_cls(self.cls, self.local_table, **self.mapper_args))