import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrangebyscore(self, name: KeyT, min: ZScoreBoundT, max: ZScoreBoundT, start: Union[int, None]=None, num: Union[int, None]=None, withscores: bool=False, score_cast_func: Union[type, Callable]=float) -> ResponseT:
    """
        Return a range of values from the sorted set ``name`` with scores
        between ``min`` and ``max``.

        If ``start`` and ``num`` are specified, then return a slice
        of the range.

        ``withscores`` indicates to return the scores along with the values.
        The return type is a list of (value, score) pairs

        `score_cast_func`` a callable used to cast the score return value

        For more information see https://redis.io/commands/zrangebyscore
        """
    if start is not None and num is None or (num is not None and start is None):
        raise DataError('``start`` and ``num`` must both be specified')
    pieces = ['ZRANGEBYSCORE', name, min, max]
    if start is not None and num is not None:
        pieces.extend(['LIMIT', start, num])
    if withscores:
        pieces.append('WITHSCORES')
    options = {'withscores': withscores, 'score_cast_func': score_cast_func}
    return self.execute_command(*pieces, **options)