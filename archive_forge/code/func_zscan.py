import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zscan(self, name: KeyT, cursor: int=0, match: Union[PatternT, None]=None, count: Union[int, None]=None, score_cast_func: Union[type, Callable]=float) -> ResponseT:
    """
        Incrementally return lists of elements in a sorted set. Also return a
        cursor indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zscan
        """
    pieces = [name, cursor]
    if match is not None:
        pieces.extend([b'MATCH', match])
    if count is not None:
        pieces.extend([b'COUNT', count])
    options = {'score_cast_func': score_cast_func}
    return self.execute_command('ZSCAN', *pieces, **options)