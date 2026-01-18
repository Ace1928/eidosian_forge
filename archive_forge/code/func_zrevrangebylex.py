import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrevrangebylex(self, name: KeyT, max: EncodableT, min: EncodableT, start: Union[int, None]=None, num: Union[int, None]=None) -> ResponseT:
    """
        Return the reversed lexicographical range of values from sorted set
        ``name`` between ``max`` and ``min``.

        If ``start`` and ``num`` are specified, then return a slice of the
        range.

        For more information see https://redis.io/commands/zrevrangebylex
        """
    if start is not None and num is None or (num is not None and start is None):
        raise DataError('``start`` and ``num`` must both be specified')
    pieces = ['ZREVRANGEBYLEX', name, max, min]
    if start is not None and num is not None:
        pieces.extend(['LIMIT', start, num])
    return self.execute_command(*pieces)