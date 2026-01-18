from __future__ import annotations
import uuid
from collections import defaultdict
from typing import Sequence
from streamlit import util
from streamlit.runtime.stats import CacheStat, group_stats
from streamlit.runtime.uploaded_file_manager import (
Return the manager's CacheStats.

        Safe to call from any thread.
        