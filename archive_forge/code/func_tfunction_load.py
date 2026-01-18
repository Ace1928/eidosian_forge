import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def tfunction_load(self, lib_code: str, replace: bool=False, config: Union[str, None]=None) -> ResponseT:
    """
        Load a new library to RedisGears.

        ``lib_code`` - the library code.
        ``config`` - a string representation of a JSON object
        that will be provided to the library on load time,
        for more information refer to
        https://github.com/RedisGears/RedisGears/blob/master/docs/function_advance_topics.md#library-configuration
        ``replace`` - an optional argument, instructs RedisGears to replace the
        function if its already exists

        For more information see https://redis.io/commands/tfunction-load/
        """
    pieces = []
    if replace:
        pieces.append('REPLACE')
    if config is not None:
        pieces.extend(['CONFIG', config])
    pieces.append(lib_code)
    return self.execute_command('TFUNCTION LOAD', *pieces)