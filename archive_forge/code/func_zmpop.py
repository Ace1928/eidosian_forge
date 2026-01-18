import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zmpop(self, num_keys: int, keys: List[str], min: Optional[bool]=False, max: Optional[bool]=False, count: Optional[int]=1) -> Union[Awaitable[list], list]:
    """
        Pop ``count`` values (default 1) off of the first non-empty sorted set
        named in the ``keys`` list.
        For more information see https://redis.io/commands/zmpop
        """
    args = [num_keys] + keys
    if min and max or (not min and (not max)):
        raise DataError
    elif min:
        args.append('MIN')
    else:
        args.append('MAX')
    if count != 1:
        args.extend(['COUNT', count])
    return self.execute_command('ZMPOP', *args)