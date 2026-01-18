import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xautoclaim(self, name: KeyT, groupname: GroupT, consumername: ConsumerT, min_idle_time: int, start_id: StreamIdT='0-0', count: Union[int, None]=None, justid: bool=False) -> ResponseT:
    """
        Transfers ownership of pending stream entries that match the specified
        criteria. Conceptually, equivalent to calling XPENDING and then XCLAIM,
        but provides a more straightforward way to deal with message delivery
        failures via SCAN-like semantics.
        name: name of the stream.
        groupname: name of the consumer group.
        consumername: name of a consumer that claims the message.
        min_idle_time: filter messages that were idle less than this amount of
        milliseconds.
        start_id: filter messages with equal or greater ID.
        count: optional integer, upper limit of the number of entries that the
        command attempts to claim. Set to 100 by default.
        justid: optional boolean, false by default. Return just an array of IDs
        of messages successfully claimed, without returning the actual message

        For more information see https://redis.io/commands/xautoclaim
        """
    try:
        if int(min_idle_time) < 0:
            raise DataError('XAUTOCLAIM min_idle_time must be a nonnegative integer')
    except TypeError:
        pass
    kwargs = {}
    pieces = [name, groupname, consumername, min_idle_time, start_id]
    try:
        if int(count) < 0:
            raise DataError('XPENDING count must be a integer >= 0')
        pieces.extend([b'COUNT', count])
    except TypeError:
        pass
    if justid:
        pieces.append(b'JUSTID')
        kwargs['parse_justid'] = True
    return self.execute_command('XAUTOCLAIM', *pieces, **kwargs)