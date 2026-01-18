import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def tfunction_delete(self, lib_name: str) -> ResponseT:
    """
        Delete a library from RedisGears.

        ``lib_name`` the library name to delete.

        For more information see https://redis.io/commands/tfunction-delete/
        """
    return self.execute_command('TFUNCTION DELETE', lib_name)