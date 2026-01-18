import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zpopmax(self, name: KeyT, count: Union[int, None]=None) -> ResponseT:
    """
        Remove and return up to ``count`` members with the highest scores
        from the sorted set ``name``.

        For more information see https://redis.io/commands/zpopmax
        """
    args = count is not None and [count] or []
    options = {'withscores': True}
    return self.execute_command('ZPOPMAX', name, *args, **options)