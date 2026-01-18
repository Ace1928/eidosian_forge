import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zremrangebyscore(self, name: KeyT, min: ZScoreBoundT, max: ZScoreBoundT) -> ResponseT:
    """
        Remove all elements in the sorted set ``name`` with scores
        between ``min`` and ``max``. Returns the number of elements removed.

        For more information see https://redis.io/commands/zremrangebyscore
        """
    return self.execute_command('ZREMRANGEBYSCORE', name, min, max)