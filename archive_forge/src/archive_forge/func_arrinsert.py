import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def arrinsert(self, name: str, path: str, index: int, *args: List[JsonType]) -> List[Union[int, None]]:
    """Insert the objects ``args`` to the array at index ``index``
        under the ``path` in key ``name``.

        For more information see `JSON.ARRINSERT <https://redis.io/commands/json.arrinsert>`_.
        """
    pieces = [name, str(path), index]
    for o in args:
        pieces.append(self._encode(o))
    return self.execute_command('JSON.ARRINSERT', *pieces)