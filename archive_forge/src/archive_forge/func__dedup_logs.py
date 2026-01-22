import time
from enum import Enum
from typing import Dict, Tuple, Union
from ray.util import PublicAPI
from ray.util.annotations import DeveloperAPI
def _dedup_logs(domain: str, value: str, repeat_after_s: int=5) -> bool:
    cur_val, ts = _log_cache_count.get(domain, (None, None))
    if value == cur_val and time.monotonic() - repeat_after_s < ts:
        return False
    else:
        _log_cache_count[domain] = (value, time.monotonic())
        return True