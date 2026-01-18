import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def tfunction_list(self, with_code: bool=False, verbose: int=0, lib_name: Union[str, None]=None) -> ResponseT:
    """
        List the functions with additional information about each function.

        ``with_code`` Show libraries code.
        ``verbose`` output verbosity level, higher number will increase verbosity level
        ``lib_name`` specifying a library name (can be used multiple times to show multiple libraries in a single command) # noqa

        For more information see https://redis.io/commands/tfunction-list/
        """
    pieces = []
    if with_code:
        pieces.append('WITHCODE')
    if verbose >= 1 and verbose <= 3:
        pieces.append('v' * verbose)
    else:
        raise DataError('verbose can be 1, 2 or 3')
    if lib_name is not None:
        pieces.append('LIBRARY')
        pieces.append(lib_name)
    return self.execute_command('TFUNCTION LIST', *pieces)