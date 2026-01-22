import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_list(self, **kwargs) -> ResponseT:
    """
        Return a list of all ACLs on the server

        For more information see https://redis.io/commands/acl-list
        """
    return self.execute_command('ACL LIST', **kwargs)