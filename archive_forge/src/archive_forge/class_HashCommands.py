import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class HashCommands(CommandsProtocol):
    """
    Redis commands for Hash data type.
    see: https://redis.io/topics/data-types-intro#redis-hashes
    """

    def hdel(self, name: str, *keys: str) -> Union[Awaitable[int], int]:
        """
        Delete ``keys`` from hash ``name``

        For more information see https://redis.io/commands/hdel
        """
        return self.execute_command('HDEL', name, *keys)

    def hexists(self, name: str, key: str) -> Union[Awaitable[bool], bool]:
        """
        Returns a boolean indicating if ``key`` exists within hash ``name``

        For more information see https://redis.io/commands/hexists
        """
        return self.execute_command('HEXISTS', name, key)

    def hget(self, name: str, key: str) -> Union[Awaitable[Optional[str]], Optional[str]]:
        """
        Return the value of ``key`` within the hash ``name``

        For more information see https://redis.io/commands/hget
        """
        return self.execute_command('HGET', name, key)

    def hgetall(self, name: str) -> Union[Awaitable[dict], dict]:
        """
        Return a Python dict of the hash's name/value pairs

        For more information see https://redis.io/commands/hgetall
        """
        return self.execute_command('HGETALL', name)

    def hincrby(self, name: str, key: str, amount: int=1) -> Union[Awaitable[int], int]:
        """
        Increment the value of ``key`` in hash ``name`` by ``amount``

        For more information see https://redis.io/commands/hincrby
        """
        return self.execute_command('HINCRBY', name, key, amount)

    def hincrbyfloat(self, name: str, key: str, amount: float=1.0) -> Union[Awaitable[float], float]:
        """
        Increment the value of ``key`` in hash ``name`` by floating ``amount``

        For more information see https://redis.io/commands/hincrbyfloat
        """
        return self.execute_command('HINCRBYFLOAT', name, key, amount)

    def hkeys(self, name: str) -> Union[Awaitable[List], List]:
        """
        Return the list of keys within hash ``name``

        For more information see https://redis.io/commands/hkeys
        """
        return self.execute_command('HKEYS', name)

    def hlen(self, name: str) -> Union[Awaitable[int], int]:
        """
        Return the number of elements in hash ``name``

        For more information see https://redis.io/commands/hlen
        """
        return self.execute_command('HLEN', name)

    def hset(self, name: str, key: Optional[str]=None, value: Optional[str]=None, mapping: Optional[dict]=None, items: Optional[list]=None) -> Union[Awaitable[int], int]:
        """
        Set ``key`` to ``value`` within hash ``name``,
        ``mapping`` accepts a dict of key/value pairs that will be
        added to hash ``name``.
        ``items`` accepts a list of key/value pairs that will be
        added to hash ``name``.
        Returns the number of fields that were added.

        For more information see https://redis.io/commands/hset
        """
        if key is None and (not mapping) and (not items):
            raise DataError("'hset' with no key value pairs")
        pieces = []
        if items:
            pieces.extend(items)
        if key is not None:
            pieces.extend((key, value))
        if mapping:
            for pair in mapping.items():
                pieces.extend(pair)
        return self.execute_command('HSET', name, *pieces)

    def hsetnx(self, name: str, key: str, value: str) -> Union[Awaitable[bool], bool]:
        """
        Set ``key`` to ``value`` within hash ``name`` if ``key`` does not
        exist.  Returns 1 if HSETNX created a field, otherwise 0.

        For more information see https://redis.io/commands/hsetnx
        """
        return self.execute_command('HSETNX', name, key, value)

    def hmset(self, name: str, mapping: dict) -> Union[Awaitable[str], str]:
        """
        Set key to value within hash ``name`` for each corresponding
        key and value from the ``mapping`` dict.

        For more information see https://redis.io/commands/hmset
        """
        warnings.warn(f'{self.__class__.__name__}.hmset() is deprecated. Use {self.__class__.__name__}.hset() instead.', DeprecationWarning, stacklevel=2)
        if not mapping:
            raise DataError("'hmset' with 'mapping' of length 0")
        items = []
        for pair in mapping.items():
            items.extend(pair)
        return self.execute_command('HMSET', name, *items)

    def hmget(self, name: str, keys: List, *args: List) -> Union[Awaitable[List], List]:
        """
        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/hmget
        """
        args = list_or_args(keys, args)
        return self.execute_command('HMGET', name, *args)

    def hvals(self, name: str) -> Union[Awaitable[List], List]:
        """
        Return the list of values within hash ``name``

        For more information see https://redis.io/commands/hvals
        """
        return self.execute_command('HVALS', name)

    def hstrlen(self, name: str, key: str) -> Union[Awaitable[int], int]:
        """
        Return the number of bytes stored in the value of ``key``
        within hash ``name``

        For more information see https://redis.io/commands/hstrlen
        """
        return self.execute_command('HSTRLEN', name, key)