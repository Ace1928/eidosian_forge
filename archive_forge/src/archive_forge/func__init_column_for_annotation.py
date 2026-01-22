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
def _init_column_for_annotation(self, cls: Type[Any], registry: _RegistryType, argument: _AnnotationScanType, originating_module: Optional[str]) -> None:
    sqltype = self.column.type
    if isinstance(argument, str) or is_fwd_ref(argument, check_generic=True):
        assert originating_module is not None
        argument = de_stringify_annotation(cls, argument, originating_module, include_generic=True)
    if is_union(argument):
        assert originating_module is not None
        argument = de_stringify_union_elements(cls, argument, originating_module)
    nullable = is_optional_union(argument)
    if not self._has_nullable:
        self.column.nullable = nullable
    our_type = de_optionalize_union_types(argument)
    use_args_from = None
    our_original_type = our_type
    if is_pep695(our_type):
        our_type = our_type.__value__
    if is_pep593(our_type):
        our_type_is_pep593 = True
        pep_593_components = typing_get_args(our_type)
        raw_pep_593_type = pep_593_components[0]
        if is_optional_union(raw_pep_593_type):
            raw_pep_593_type = de_optionalize_union_types(raw_pep_593_type)
            nullable = True
            if not self._has_nullable:
                self.column.nullable = nullable
        for elem in pep_593_components[1:]:
            if isinstance(elem, MappedColumn):
                use_args_from = elem
                break
    else:
        our_type_is_pep593 = False
        raw_pep_593_type = None
    if use_args_from is not None:
        if not self._has_insert_default and use_args_from.column.default is not None:
            self.column.default = None
        use_args_from.column._merge(self.column)
        sqltype = self.column.type
        if use_args_from.deferred is not _NoArg.NO_ARG and self.deferred is _NoArg.NO_ARG:
            self.deferred = use_args_from.deferred
        if use_args_from.deferred_group is not None and self.deferred_group is None:
            self.deferred_group = use_args_from.deferred_group
        if use_args_from.deferred_raiseload is not None and self.deferred_raiseload is None:
            self.deferred_raiseload = use_args_from.deferred_raiseload
        if use_args_from._use_existing_column and (not self._use_existing_column):
            self._use_existing_column = True
        if use_args_from.active_history:
            self.active_history = use_args_from.active_history
        if use_args_from._sort_order is not None and self._sort_order is _NoArg.NO_ARG:
            self._sort_order = use_args_from._sort_order
        if use_args_from.column.key is not None or use_args_from.column.name is not None:
            util.warn_deprecated("Can't use the 'key' or 'name' arguments in Annotated with mapped_column(); this will be ignored", '2.0.22')
        if use_args_from._has_dataclass_arguments:
            for idx, arg in enumerate(use_args_from._attribute_options._fields):
                if use_args_from._attribute_options[idx] is not _NoArg.NO_ARG:
                    arg = arg.replace('dataclasses_', '')
                    util.warn_deprecated(f"Argument '{arg}' is a dataclass argument and cannot be specified within a mapped_column() bundled inside of an Annotated object", '2.0.22')
    if sqltype._isnull and (not self.column.foreign_keys):
        new_sqltype = None
        if our_type_is_pep593:
            checks = [our_original_type, raw_pep_593_type]
        else:
            checks = [our_original_type]
        for check_type in checks:
            new_sqltype = registry._resolve_type(check_type)
            if new_sqltype is not None:
                break
        else:
            if isinstance(our_type, TypeEngine) or (isinstance(our_type, type) and issubclass(our_type, TypeEngine)):
                raise sa_exc.ArgumentError(f'The type provided inside the {self.column.key!r} attribute Mapped annotation is the SQLAlchemy type {our_type}. Expected a Python type instead')
            else:
                raise sa_exc.ArgumentError(f'Could not locate SQLAlchemy Core type for Python type {our_type} inside the {self.column.key!r} attribute Mapped annotation')
        self.column._set_type(new_sqltype)