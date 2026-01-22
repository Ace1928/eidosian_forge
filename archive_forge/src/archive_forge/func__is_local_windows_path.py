import pathlib
import sys
import urllib
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from ray.data._internal.util import _resolve_custom_scheme
def _is_local_windows_path(path: str) -> bool:
    """Determines if path is a Windows file-system location."""
    if sys.platform != 'win32':
        return False
    if len(path) >= 1 and path[0] == '\\':
        return True
    if len(path) >= 3 and path[1] == ':' and (path[2] == '/' or path[2] == '\\') and path[0].isalpha():
        return True
    return False