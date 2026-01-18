import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zdiffstore(self, dest: KeyT, keys: KeysT) -> ResponseT:
    """
        Computes the difference between the first and all successive input
        sorted sets provided in ``keys`` and stores the result in ``dest``.

        For more information see https://redis.io/commands/zdiffstore
        """
    pieces = [len(keys), *keys]
    return self.execute_command('ZDIFFSTORE', dest, *pieces)