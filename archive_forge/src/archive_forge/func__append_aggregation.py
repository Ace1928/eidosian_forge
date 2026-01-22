from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_aggregation(params: List[str], aggregation_type: Optional[str], bucket_size_msec: Optional[int]):
    """Append AGGREGATION property to params."""
    if aggregation_type is not None:
        params.extend(['AGGREGATION', aggregation_type, bucket_size_msec])