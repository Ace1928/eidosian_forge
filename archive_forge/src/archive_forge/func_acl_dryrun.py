import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_dryrun(self, username, *args, **kwargs):
    """
        Simulate the execution of a given command by a given ``username``.

        For more information see https://redis.io/commands/acl-dryrun
        """
    return self.execute_command('ACL DRYRUN', username, *args, **kwargs)