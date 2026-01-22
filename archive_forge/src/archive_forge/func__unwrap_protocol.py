import pathlib
import sys
import urllib
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from ray.data._internal.util import _resolve_custom_scheme
def _unwrap_protocol(path):
    """
    Slice off any protocol prefixes on path.
    """
    if sys.platform == 'win32' and _is_local_windows_path(path):
        return pathlib.Path(path).as_posix()
    parsed = urllib.parse.urlparse(path, allow_fragments=False)
    query = '?' + parsed.query if parsed.query else ''
    netloc = parsed.netloc
    if parsed.scheme == 's3' and '@' in parsed.netloc:
        netloc = parsed.netloc.split('@')[-1]
    parsed_path = parsed.path
    if sys.platform == 'win32' and (not netloc) and (len(parsed_path) >= 3) and (parsed_path[0] == '/') and parsed_path[1].isalpha() and (parsed_path[2:4] in (':', ':/')):
        parsed_path = parsed_path[1:]
    return netloc + parsed_path + query