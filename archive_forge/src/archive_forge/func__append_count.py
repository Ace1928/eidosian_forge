from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_count(params: List[str], count: Optional[int]):
    """Append COUNT property to params."""
    if count is not None:
        params.extend(['COUNT', count])