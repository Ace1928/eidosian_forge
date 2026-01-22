from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_timestamp(params: List[str], timestamp: Optional[int]):
    """Append TIMESTAMP property to params."""
    if timestamp is not None:
        params.extend(['TIMESTAMP', timestamp])