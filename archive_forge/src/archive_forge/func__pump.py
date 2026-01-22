import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _pump(self, from_file, to_file):
    """Most children will need to copy from one file-like
        object or string to another one.
        This just gives them something easy to call.
        """
    return osutils.pumpfile(from_file, to_file)