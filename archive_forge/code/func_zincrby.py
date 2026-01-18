import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zincrby(self, name: KeyT, amount: float, value: EncodableT) -> ResponseT:
    """
        Increment the score of ``value`` in sorted set ``name`` by ``amount``

        For more information see https://redis.io/commands/zincrby
        """
    return self.execute_command('ZINCRBY', name, amount, value)