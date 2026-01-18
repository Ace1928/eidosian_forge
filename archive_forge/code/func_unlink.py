import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def unlink(self, *names: KeyT) -> ResponseT:
    """
        Unlink one or more keys specified by ``names``

        For more information see https://redis.io/commands/unlink
        """
    return self.execute_command('UNLINK', *names)