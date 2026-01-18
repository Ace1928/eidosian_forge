import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zlexcount(self, name, min, max):
    """
        Return the number of items in the sorted set ``name`` between the
        lexicographical range ``min`` and ``max``.

        For more information see https://redis.io/commands/zlexcount
        """
    return self.execute_command('ZLEXCOUNT', name, min, max)