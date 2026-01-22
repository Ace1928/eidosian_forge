from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
Check if a Python object is JSON-serializable.

    obj: The object to check.
    RETURNS (bool): Whether the object is JSON-serializable.
    