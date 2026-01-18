import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xreadgroup(self, groupname: str, consumername: str, streams: Dict[KeyT, StreamIdT], count: Union[int, None]=None, block: Union[int, None]=None, noack: bool=False) -> ResponseT:
    """
        Read from a stream via a consumer group.

        groupname: name of the consumer group.

        consumername: name of the requesting consumer.

        streams: a dict of stream names to stream IDs, where
               IDs indicate the last ID already seen.

        count: if set, only return this many items, beginning with the
               earliest available.

        block: number of milliseconds to wait, if nothing already present.
        noack: do not add messages to the PEL

        For more information see https://redis.io/commands/xreadgroup
        """
    pieces: list[EncodableT] = [b'GROUP', groupname, consumername]
    if count is not None:
        if not isinstance(count, int) or count < 1:
            raise DataError('XREADGROUP count must be a positive integer')
        pieces.append(b'COUNT')
        pieces.append(str(count))
    if block is not None:
        if not isinstance(block, int) or block < 0:
            raise DataError('XREADGROUP block must be a non-negative integer')
        pieces.append(b'BLOCK')
        pieces.append(str(block))
    if noack:
        pieces.append(b'NOACK')
    if not isinstance(streams, dict) or len(streams) == 0:
        raise DataError('XREADGROUP streams must be a non empty dict')
    pieces.append(b'STREAMS')
    pieces.extend(streams.keys())
    pieces.extend(streams.values())
    return self.execute_command('XREADGROUP', *pieces)