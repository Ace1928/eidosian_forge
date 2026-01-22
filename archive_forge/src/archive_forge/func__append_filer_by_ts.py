from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
@staticmethod
def _append_filer_by_ts(params: List[str], ts_list: Optional[List[int]]):
    """Append FILTER_BY_TS property to params."""
    if ts_list is not None:
        params.extend(['FILTER_BY_TS', *ts_list])