import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _try_transport_factories(base, factory_list):
    last_err = None
    for factory in factory_list:
        try:
            return (factory.get_obj()(base), None)
        except errors.DependencyNotPresent as e:
            mutter('failed to instantiate transport %r for %r: %r' % (factory, base, e))
            last_err = e
            continue
    return (None, last_err)