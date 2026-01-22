import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class HyperlogCommands(CommandsProtocol):
    """
    Redis commands of HyperLogLogs data type.
    see: https://redis.io/topics/data-types-intro#hyperloglogs
    """

    def pfadd(self, name: KeyT, *values: FieldT) -> ResponseT:
        """
        Adds the specified elements to the specified HyperLogLog.

        For more information see https://redis.io/commands/pfadd
        """
        return self.execute_command('PFADD', name, *values)

    def pfcount(self, *sources: KeyT) -> ResponseT:
        """
        Return the approximated cardinality of
        the set observed by the HyperLogLog at key(s).

        For more information see https://redis.io/commands/pfcount
        """
        return self.execute_command('PFCOUNT', *sources)

    def pfmerge(self, dest: KeyT, *sources: KeyT) -> ResponseT:
        """
        Merge N different HyperLogLogs into a single one.

        For more information see https://redis.io/commands/pfmerge
        """
        return self.execute_command('PFMERGE', dest, *sources)