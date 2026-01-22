import bisect
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional
import numpy as np
import ray
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockAccessor
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def _find_le(self, x: Any) -> int:
    i = bisect.bisect_left(self._upper_bounds, x)
    if i >= len(self._upper_bounds) or x < self._lower_bound:
        return None
    return i