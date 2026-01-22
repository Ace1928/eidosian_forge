from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def __range_params(self, key: KeyT, from_time: Union[int, str], to_time: Union[int, str], count: Optional[int], aggregation_type: Optional[str], bucket_size_msec: Optional[int], filter_by_ts: Optional[List[int]], filter_by_min_value: Optional[int], filter_by_max_value: Optional[int], align: Optional[Union[int, str]], latest: Optional[bool], bucket_timestamp: Optional[str], empty: Optional[bool]):
    """Create TS.RANGE and TS.REVRANGE arguments."""
    params = [key, from_time, to_time]
    self._append_latest(params, latest)
    self._append_filer_by_ts(params, filter_by_ts)
    self._append_filer_by_value(params, filter_by_min_value, filter_by_max_value)
    self._append_count(params, count)
    self._append_align(params, align)
    self._append_aggregation(params, aggregation_type, bucket_size_msec)
    self._append_bucket_timestamp(params, bucket_timestamp)
    self._append_empty(params, empty)
    return params