import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional
from ray.serve._private.constants import SERVE_LOGGER_NAME
def _get_datapoints(self, key: str, window_start_timestamp_s: float) -> List[float]:
    """Get all data points given key after window_start_timestamp_s"""
    datapoints = self.data[key]
    idx = bisect.bisect(a=datapoints, x=TimeStampedValue(timestamp=window_start_timestamp_s, value=0))
    return datapoints[idx:]