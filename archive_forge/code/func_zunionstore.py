import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def zunionstore(self, dest: KeyT, keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]], aggregate: Union[str, None]=None) -> ResponseT:
    """
        Union multiple sorted sets specified by ``keys`` into
        a new sorted set, ``dest``. Scores in the destination will be
        aggregated based on the ``aggregate``, or SUM if none is provided.

        For more information see https://redis.io/commands/zunionstore
        """
    return self._zaggregate('ZUNIONSTORE', dest, keys, aggregate)