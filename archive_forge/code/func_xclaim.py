import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xclaim(self, name: KeyT, groupname: GroupT, consumername: ConsumerT, min_idle_time: int, message_ids: Union[List[StreamIdT], Tuple[StreamIdT]], idle: Union[int, None]=None, time: Union[int, None]=None, retrycount: Union[int, None]=None, force: bool=False, justid: bool=False) -> ResponseT:
    """
        Changes the ownership of a pending message.

        name: name of the stream.

        groupname: name of the consumer group.

        consumername: name of a consumer that claims the message.

        min_idle_time: filter messages that were idle less than this amount of
        milliseconds

        message_ids: non-empty list or tuple of message IDs to claim

        idle: optional. Set the idle time (last time it was delivered) of the
        message in ms

        time: optional integer. This is the same as idle but instead of a
        relative amount of milliseconds, it sets the idle time to a specific
        Unix time (in milliseconds).

        retrycount: optional integer. set the retry counter to the specified
        value. This counter is incremented every time a message is delivered
        again.

        force: optional boolean, false by default. Creates the pending message
        entry in the PEL even if certain specified IDs are not already in the
        PEL assigned to a different client.

        justid: optional boolean, false by default. Return just an array of IDs
        of messages successfully claimed, without returning the actual message

        For more information see https://redis.io/commands/xclaim
        """
    if not isinstance(min_idle_time, int) or min_idle_time < 0:
        raise DataError('XCLAIM min_idle_time must be a non negative integer')
    if not isinstance(message_ids, (list, tuple)) or not message_ids:
        raise DataError('XCLAIM message_ids must be a non empty list or tuple of message IDs to claim')
    kwargs = {}
    pieces: list[EncodableT] = [name, groupname, consumername, str(min_idle_time)]
    pieces.extend(list(message_ids))
    if idle is not None:
        if not isinstance(idle, int):
            raise DataError('XCLAIM idle must be an integer')
        pieces.extend((b'IDLE', str(idle)))
    if time is not None:
        if not isinstance(time, int):
            raise DataError('XCLAIM time must be an integer')
        pieces.extend((b'TIME', str(time)))
    if retrycount is not None:
        if not isinstance(retrycount, int):
            raise DataError('XCLAIM retrycount must be an integer')
        pieces.extend((b'RETRYCOUNT', str(retrycount)))
    if force:
        if not isinstance(force, bool):
            raise DataError('XCLAIM force must be a boolean')
        pieces.append(b'FORCE')
    if justid:
        if not isinstance(justid, bool):
            raise DataError('XCLAIM justid must be a boolean')
        pieces.append(b'JUSTID')
        kwargs['parse_justid'] = True
    return self.execute_command('XCLAIM', *pieces, **kwargs)