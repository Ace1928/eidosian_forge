import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def arrlen(self, name: str, path: Optional[str]=Path.root_path()) -> List[Union[int, None]]:
    """Return the length of the array JSON value under ``path``
        at key``name``.

        For more information see `JSON.ARRLEN <https://redis.io/commands/json.arrlen>`_.
        """
    return self.execute_command('JSON.ARRLEN', name, str(path))