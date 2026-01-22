from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import interfaces
from .descriptor_props import SynonymProperty
from .properties import ColumnProperty
from .util import class_mapper
from .. import exc
from .. import inspection
from .. import util
from ..sql.schema import _get_table_key
from ..util.typing import CallableReference
def _access_cls(self, key: str) -> Any:
    cls = self.cls
    manager = attributes.manager_of_class(cls)
    decl_base = manager.registry
    assert decl_base is not None
    decl_class_registry = decl_base._class_registry
    metadata = decl_base.metadata
    if self.favor_tables:
        if key in metadata.tables:
            return metadata.tables[key]
        elif key in metadata._schemas:
            return _GetTable(key, getattr(cls, 'metadata', metadata))
    if key in decl_class_registry:
        return _determine_container(key, decl_class_registry[key])
    if not self.favor_tables:
        if key in metadata.tables:
            return metadata.tables[key]
        elif key in metadata._schemas:
            return _GetTable(key, getattr(cls, 'metadata', metadata))
    if '_sa_module_registry' in decl_class_registry and key in cast(_ModuleMarker, decl_class_registry['_sa_module_registry']):
        registry = cast(_ModuleMarker, decl_class_registry['_sa_module_registry'])
        return registry.resolve_attr(key)
    elif self._resolvers:
        for resolv in self._resolvers:
            value = resolv(key)
            if value is not None:
                return value
    return self.fallback[key]