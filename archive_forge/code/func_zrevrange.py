import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrevrange(self, name: KeyT, start: int, end: int, withscores: bool=False, score_cast_func: Union[type, Callable]=float) -> ResponseT:
    """
        Return a range of values from sorted set ``name`` between
        ``start`` and ``end`` sorted in descending order.

        ``start`` and ``end`` can be negative, indicating the end of the range.

        ``withscores`` indicates to return the scores along with the values
        The return type is a list of (value, score) pairs

        ``score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zrevrange
        """
    pieces = ['ZREVRANGE', name, start, end]
    if withscores:
        pieces.append(b'WITHSCORES')
    options = {'withscores': withscores, 'score_cast_func': score_cast_func}
    return self.execute_command(*pieces, **options)