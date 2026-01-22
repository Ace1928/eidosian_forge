from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def _array_to_buffer(array: List[float], dtype: Any=np.float32) -> bytes:
    return np.array(array).astype(dtype).tobytes()