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
class _MultipleClassMarker(ClsRegistryToken):
    """refers to multiple classes of the same name
    within _decl_class_registry.

    """
    __slots__ = ('on_remove', 'contents', '__weakref__')
    contents: Set[weakref.ref[Type[Any]]]
    on_remove: CallableReference[Optional[Callable[[], None]]]

    def __init__(self, classes: Iterable[Type[Any]], on_remove: Optional[Callable[[], None]]=None):
        self.on_remove = on_remove
        self.contents = {weakref.ref(item, self._remove_item) for item in classes}
        _registries.add(self)

    def remove_item(self, cls: Type[Any]) -> None:
        self._remove_item(weakref.ref(cls))

    def __iter__(self) -> Generator[Optional[Type[Any]], None, None]:
        return (ref() for ref in self.contents)

    def attempt_get(self, path: List[str], key: str) -> Type[Any]:
        if len(self.contents) > 1:
            raise exc.InvalidRequestError('Multiple classes found for path "%s" in the registry of this declarative base. Please use a fully module-qualified path.' % '.'.join(path + [key]))
        else:
            ref = list(self.contents)[0]
            cls = ref()
            if cls is None:
                raise NameError(key)
            return cls

    def _remove_item(self, ref: weakref.ref[Type[Any]]) -> None:
        self.contents.discard(ref)
        if not self.contents:
            _registries.discard(self)
            if self.on_remove:
                self.on_remove()

    def add_item(self, item: Type[Any]) -> None:
        modules = {cls.__module__ for cls in [ref() for ref in list(self.contents)] if cls is not None}
        if item.__module__ in modules:
            util.warn('This declarative base already contains a class with the same class name and module name as %s.%s, and will be replaced in the string-lookup table.' % (item.__module__, item.__name__))
        self.contents.add(weakref.ref(item, self._remove_item))