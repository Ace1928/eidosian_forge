import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _get_total(self, multi):
    """Try to figure out how many entries are in multi,
        but if not possible, return None.
        """
    try:
        return len(multi)
    except TypeError:
        return None