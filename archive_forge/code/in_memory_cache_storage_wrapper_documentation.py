from __future__ import annotations
import math
import threading
from cachetools import TTLCache
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.storage.cache_storage_protocol import (
from streamlit.runtime.stats import CacheStat
Closes the cache storage