import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
def arrpop(self, name: str, path: Optional[str]=Path.root_path(), index: Optional[int]=-1) -> List[Union[str, None]]:
    """Pop the element at ``index`` in the array JSON value under
        ``path`` at key ``name``.

        For more information see `JSON.ARRPOP <https://redis.io/commands/json.arrpop>`_.
        """
    return self.execute_command('JSON.ARRPOP', name, str(path), index)