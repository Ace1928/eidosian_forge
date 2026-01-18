import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zrank(self, name: KeyT, value: EncodableT, withscore: bool=False) -> ResponseT:
    """
        Returns a 0-based value indicating the rank of ``value`` in sorted set
        ``name``.
        The optional WITHSCORE argument supplements the command's
        reply with the score of the element returned.

        For more information see https://redis.io/commands/zrank
        """
    if withscore:
        return self.execute_command('ZRANK', name, value, 'WITHSCORE')
    return self.execute_command('ZRANK', name, value)