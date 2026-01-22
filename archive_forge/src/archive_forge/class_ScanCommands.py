import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class ScanCommands(CommandsProtocol):
    """
    Redis SCAN commands.
    see: https://redis.io/commands/scan
    """

    def scan(self, cursor: int=0, match: Union[PatternT, None]=None, count: Union[int, None]=None, _type: Union[str, None]=None, **kwargs) -> ResponseT:
        """
        Incrementally return lists of key names. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.

        For more information see https://redis.io/commands/scan
        """
        pieces: list[EncodableT] = [cursor]
        if match is not None:
            pieces.extend([b'MATCH', match])
        if count is not None:
            pieces.extend([b'COUNT', count])
        if _type is not None:
            pieces.extend([b'TYPE', _type])
        return self.execute_command('SCAN', *pieces, **kwargs)

    def scan_iter(self, match: Union[PatternT, None]=None, count: Union[int, None]=None, _type: Union[str, None]=None, **kwargs) -> Iterator:
        """
        Make an iterator using the SCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = self.scan(cursor=cursor, match=match, count=count, _type=_type, **kwargs)
            yield from data

    def sscan(self, name: KeyT, cursor: int=0, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> ResponseT:
        """
        Incrementally return lists of elements in a set. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        For more information see https://redis.io/commands/sscan
        """
        pieces: list[EncodableT] = [name, cursor]
        if match is not None:
            pieces.extend([b'MATCH', match])
        if count is not None:
            pieces.extend([b'COUNT', count])
        return self.execute_command('SSCAN', *pieces)

    def sscan_iter(self, name: KeyT, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> Iterator:
        """
        Make an iterator using the SSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = self.sscan(name, cursor=cursor, match=match, count=count)
            yield from data

    def hscan(self, name: KeyT, cursor: int=0, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> ResponseT:
        """
        Incrementally return key/value slices in a hash. Also return a cursor
        indicating the scan position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        For more information see https://redis.io/commands/hscan
        """
        pieces: list[EncodableT] = [name, cursor]
        if match is not None:
            pieces.extend([b'MATCH', match])
        if count is not None:
            pieces.extend([b'COUNT', count])
        return self.execute_command('HSCAN', *pieces)

    def hscan_iter(self, name: str, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> Iterator:
        """
        Make an iterator using the HSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = self.hscan(name, cursor=cursor, match=match, count=count)
            yield from data.items()

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

    def zscan_iter(self, name: KeyT, match: Union[PatternT, None]=None, count: Union[int, None]=None, score_cast_func: Union[type, Callable]=float) -> Iterator:
        """
        Make an iterator using the ZSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = self.zscan(name, cursor=cursor, match=match, count=count, score_cast_func=score_cast_func)
            yield from data