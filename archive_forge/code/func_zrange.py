import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrange(self, name: KeyT, start: int, end: int, desc: bool=False, withscores: bool=False, score_cast_func: Union[type, Callable]=float, byscore: bool=False, bylex: bool=False, offset: int=None, num: int=None) -> ResponseT:
    """
        Return a range of values from sorted set ``name`` between
        ``start`` and ``end`` sorted in ascending order.

        ``start`` and ``end`` can be negative, indicating the end of the range.

        ``desc`` a boolean indicating whether to sort the results in reversed
        order.

        ``withscores`` indicates to return the scores along with the values.
        The return type is a list of (value, score) pairs.

        ``score_cast_func`` a callable used to cast the score return value.

        ``byscore`` when set to True, returns the range of elements from the
        sorted set having scores equal or between ``start`` and ``end``.

        ``bylex`` when set to True, returns the range of elements from the
        sorted set between the ``start`` and ``end`` lexicographical closed
        range intervals.
        Valid ``start`` and ``end`` must start with ( or [, in order to specify
        whether the range interval is exclusive or inclusive, respectively.

        ``offset`` and ``num`` are specified, then return a slice of the range.
        Can't be provided when using ``bylex``.

        For more information see https://redis.io/commands/zrange
        """
    if not byscore and (not bylex) and (offset is None and num is None) and desc:
        return self.zrevrange(name, start, end, withscores, score_cast_func)
    return self._zrange('ZRANGE', None, name, start, end, desc, byscore, bylex, withscores, score_cast_func, offset, num)