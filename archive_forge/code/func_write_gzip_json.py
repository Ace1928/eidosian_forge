from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def write_gzip_json(path: FilePath, data: JSONInput, indent: int=2) -> None:
    """Create a .json.gz file and dump contents.

    path (FilePath): The file path.
    data (JSONInput): The JSON-serializable data to output.
    indent (int): Number of spaces used to indent JSON.
    """
    json_data = json_dumps(data, indent=indent)
    file_path = force_string(path)
    with gzip.open(file_path, 'w') as f:
        f.write(json_data.encode('utf-8'))