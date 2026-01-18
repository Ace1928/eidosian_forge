import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def waitaof(self, num_local: int, num_replicas: int, timeout: int, **kwargs) -> ResponseT:
    """
        This command blocks the current client until all previous write
        commands by that client are acknowledged as having been fsynced
        to the AOF of the local Redis and/or at least the specified number
        of replicas.

        For more information see https://redis.io/commands/waitaof
        """
    return self.execute_command('WAITAOF', num_local, num_replicas, timeout, **kwargs)