import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrem(self, name: KeyT, *values: FieldT) -> ResponseT:
    """
        Remove member ``values`` from sorted set ``name``

        For more information see https://redis.io/commands/zrem
        """
    return self.execute_command('ZREM', name, *values)