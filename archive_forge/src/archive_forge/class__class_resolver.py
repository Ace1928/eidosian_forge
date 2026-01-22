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
class _class_resolver:
    __slots__ = ('cls', 'prop', 'arg', 'fallback', '_dict', '_resolvers', 'favor_tables')
    cls: Type[Any]
    prop: RelationshipProperty[Any]
    fallback: Mapping[str, Any]
    arg: str
    favor_tables: bool
    _resolvers: Tuple[Callable[[str], Any], ...]

    def __init__(self, cls: Type[Any], prop: RelationshipProperty[Any], fallback: Mapping[str, Any], arg: str, favor_tables: bool=False):
        self.cls = cls
        self.prop = prop
        self.arg = arg
        self.fallback = fallback
        self._dict = util.PopulateDict(self._access_cls)
        self._resolvers = ()
        self.favor_tables = favor_tables

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

    def _raise_for_name(self, name: str, err: Exception) -> NoReturn:
        generic_match = re.match('(.+)\\[(.+)\\]', name)
        if generic_match:
            clsarg = generic_match.group(2).strip("'")
            raise exc.InvalidRequestError(f'''When initializing mapper {self.prop.parent}, expression "relationship({self.arg!r})" seems to be using a generic class as the argument to relationship(); please state the generic argument using an annotation, e.g. "{self.prop.key}: Mapped[{generic_match.group(1)}['{clsarg}']] = relationship()"''') from err
        else:
            raise exc.InvalidRequestError('When initializing mapper %s, expression %r failed to locate a name (%r). If this is a class name, consider adding this relationship() to the %r class after both dependent classes have been defined.' % (self.prop.parent, self.arg, name, self.cls)) from err

    def _resolve_name(self) -> Union[Table, Type[Any], _ModNS]:
        name = self.arg
        d = self._dict
        rval = None
        try:
            for token in name.split('.'):
                if rval is None:
                    rval = d[token]
                else:
                    rval = getattr(rval, token)
        except KeyError as err:
            self._raise_for_name(name, err)
        except NameError as n:
            self._raise_for_name(n.args[0], n)
        else:
            if isinstance(rval, _GetColumns):
                return rval.cls
            else:
                if TYPE_CHECKING:
                    assert isinstance(rval, (type, Table, _ModNS))
                return rval

    def __call__(self) -> Any:
        try:
            x = eval(self.arg, globals(), self._dict)
            if isinstance(x, _GetColumns):
                return x.cls
            else:
                return x
        except NameError as n:
            self._raise_for_name(n.args[0], n)