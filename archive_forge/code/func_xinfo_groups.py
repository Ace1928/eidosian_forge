import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def xinfo_groups(self, name: KeyT) -> ResponseT:
    """
        Returns general information about the consumer groups of the stream.
        name: name of the stream.

        For more information see https://redis.io/commands/xinfo-groups
        """
    return self.execute_command('XINFO GROUPS', name)