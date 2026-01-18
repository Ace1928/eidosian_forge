import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def unwatch(self) -> None:
    """
        Unwatches the value at key ``name``, or None of the key doesn't exist

        For more information see https://redis.io/commands/unwatch
        """
    warnings.warn(DeprecationWarning('Call UNWATCH from a Pipeline object'))