import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def unregister_transport(scheme, factory):
    """Unregister a transport."""
    l = transport_list_registry.get(scheme)
    for i in l:
        o = i.get_obj()
        if o == factory:
            transport_list_registry.get(scheme).remove(i)
            break
    if len(l) == 0:
        transport_list_registry.remove(scheme)