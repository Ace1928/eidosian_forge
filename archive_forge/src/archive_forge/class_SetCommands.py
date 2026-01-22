import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class SetCommands(CommandsProtocol):
    """
    Redis commands for Set data type.
    see: https://redis.io/topics/data-types#sets
    """

    def sadd(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Add ``value(s)`` to set ``name``

        For more information see https://redis.io/commands/sadd
        """
        return self.execute_command('SADD', name, *values)

    def scard(self, name: str) -> Union[Awaitable[int], int]:
        """
        Return the number of elements in set ``name``

        For more information see https://redis.io/commands/scard
        """
        return self.execute_command('SCARD', name)

    def sdiff(self, keys: List, *args: List) -> Union[Awaitable[list], list]:
        """
        Return the difference of sets specified by ``keys``

        For more information see https://redis.io/commands/sdiff
        """
        args = list_or_args(keys, args)
        return self.execute_command('SDIFF', *args)

    def sdiffstore(self, dest: str, keys: List, *args: List) -> Union[Awaitable[int], int]:
        """
        Store the difference of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sdiffstore
        """
        args = list_or_args(keys, args)
        return self.execute_command('SDIFFSTORE', dest, *args)

    def sinter(self, keys: List, *args: List) -> Union[Awaitable[list], list]:
        """
        Return the intersection of sets specified by ``keys``

        For more information see https://redis.io/commands/sinter
        """
        args = list_or_args(keys, args)
        return self.execute_command('SINTER', *args)

    def sintercard(self, numkeys: int, keys: List[str], limit: int=0) -> Union[Awaitable[int], int]:
        """
        Return the cardinality of the intersect of multiple sets specified by ``keys`.

        When LIMIT provided (defaults to 0 and means unlimited), if the intersection
        cardinality reaches limit partway through the computation, the algorithm will
        exit and yield limit as the cardinality

        For more information see https://redis.io/commands/sintercard
        """
        args = [numkeys, *keys, 'LIMIT', limit]
        return self.execute_command('SINTERCARD', *args)

    def sinterstore(self, dest: str, keys: List, *args: List) -> Union[Awaitable[int], int]:
        """
        Store the intersection of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sinterstore
        """
        args = list_or_args(keys, args)
        return self.execute_command('SINTERSTORE', dest, *args)

    def sismember(self, name: str, value: str) -> Union[Awaitable[Union[Literal[0], Literal[1]]], Union[Literal[0], Literal[1]]]:
        """
        Return whether ``value`` is a member of set ``name``:
        - 1 if the value is a member of the set.
        - 0 if the value is not a member of the set or if key does not exist.

        For more information see https://redis.io/commands/sismember
        """
        return self.execute_command('SISMEMBER', name, value)

    def smembers(self, name: str) -> Union[Awaitable[Set], Set]:
        """
        Return all members of the set ``name``

        For more information see https://redis.io/commands/smembers
        """
        return self.execute_command('SMEMBERS', name)

    def smismember(self, name: str, values: List, *args: List) -> Union[Awaitable[List[Union[Literal[0], Literal[1]]]], List[Union[Literal[0], Literal[1]]]]:
        """
        Return whether each value in ``values`` is a member of the set ``name``
        as a list of ``int`` in the order of ``values``:
        - 1 if the value is a member of the set.
        - 0 if the value is not a member of the set or if key does not exist.

        For more information see https://redis.io/commands/smismember
        """
        args = list_or_args(values, args)
        return self.execute_command('SMISMEMBER', name, *args)

    def smove(self, src: str, dst: str, value: str) -> Union[Awaitable[bool], bool]:
        """
        Move ``value`` from set ``src`` to set ``dst`` atomically

        For more information see https://redis.io/commands/smove
        """
        return self.execute_command('SMOVE', src, dst, value)

    def spop(self, name: str, count: Optional[int]=None) -> Union[str, List, None]:
        """
        Remove and return a random member of set ``name``

        For more information see https://redis.io/commands/spop
        """
        args = count is not None and [count] or []
        return self.execute_command('SPOP', name, *args)

    def srandmember(self, name: str, number: Optional[int]=None) -> Union[str, List, None]:
        """
        If ``number`` is None, returns a random member of set ``name``.

        If ``number`` is supplied, returns a list of ``number`` random
        members of set ``name``. Note this is only available when running
        Redis 2.6+.

        For more information see https://redis.io/commands/srandmember
        """
        args = number is not None and [number] or []
        return self.execute_command('SRANDMEMBER', name, *args)

    def srem(self, name: str, *values: FieldT) -> Union[Awaitable[int], int]:
        """
        Remove ``values`` from set ``name``

        For more information see https://redis.io/commands/srem
        """
        return self.execute_command('SREM', name, *values)

    def sunion(self, keys: List, *args: List) -> Union[Awaitable[List], List]:
        """
        Return the union of sets specified by ``keys``

        For more information see https://redis.io/commands/sunion
        """
        args = list_or_args(keys, args)
        return self.execute_command('SUNION', *args)

    def sunionstore(self, dest: str, keys: List, *args: List) -> Union[Awaitable[int], int]:
        """
        Store the union of sets specified by ``keys`` into a new
        set named ``dest``.  Returns the number of keys in the new set.

        For more information see https://redis.io/commands/sunionstore
        """
        args = list_or_args(keys, args)
        return self.execute_command('SUNIONSTORE', dest, *args)