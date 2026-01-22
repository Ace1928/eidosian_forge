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
def _key_is_empty(key: str, decl_class_registry: _ClsRegistryType, test: Callable[[Any], bool]) -> bool:
    """test if a key is empty of a certain object.

    used for unit tests against the registry to see if garbage collection
    is working.

    "test" is a callable that will be passed an object should return True
    if the given object is the one we were looking for.

    We can't pass the actual object itself b.c. this is for testing garbage
    collection; the caller will have to have removed references to the
    object itself.

    """
    if key not in decl_class_registry:
        return True
    thing = decl_class_registry[key]
    if isinstance(thing, _MultipleClassMarker):
        for sub_thing in thing.contents:
            if test(sub_thing):
                return False
        else:
            raise NotImplementedError('unknown codepath')
    else:
        return not test(thing)