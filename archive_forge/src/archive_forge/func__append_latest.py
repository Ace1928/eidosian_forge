from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_latest(params: List[str], latest: Optional[bool]):
    """Append LATEST property to params."""
    if latest:
        params.append('LATEST')