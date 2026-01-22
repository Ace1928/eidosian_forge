from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_filer_by_value(params: List[str], min_value: Optional[int], max_value: Optional[int]):
    """Append FILTER_BY_VALUE property to params."""
    if min_value is not None and max_value is not None:
        params.extend(['FILTER_BY_VALUE', min_value, max_value])