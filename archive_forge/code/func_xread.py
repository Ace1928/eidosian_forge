import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xread(self, streams: Dict[KeyT, StreamIdT], count: Union[int, None]=None, block: Union[int, None]=None) -> ResponseT:
    """
        Block and monitor multiple streams for new data.

        streams: a dict of stream names to stream IDs, where
                   IDs indicate the last ID already seen.

        count: if set, only return this many items, beginning with the
               earliest available.

        block: number of milliseconds to wait, if nothing already present.

        For more information see https://redis.io/commands/xread
        """
    pieces = []
    if block is not None:
        if not isinstance(block, int) or block < 0:
            raise DataError('XREAD block must be a non-negative integer')
        pieces.append(b'BLOCK')
        pieces.append(str(block))
    if count is not None:
        if not isinstance(count, int) or count < 1:
            raise DataError('XREAD count must be a positive integer')
        pieces.append(b'COUNT')
        pieces.append(str(count))
    if not isinstance(streams, dict) or len(streams) == 0:
        raise DataError('XREAD streams must be a non empty dict')
    pieces.append(b'STREAMS')
    keys, values = zip(*streams.items())
    pieces.extend(keys)
    pieces.extend(values)
    return self.execute_command('XREAD', *pieces)