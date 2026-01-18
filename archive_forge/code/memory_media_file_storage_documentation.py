from __future__ import annotations
import contextlib
import hashlib
import mimetypes
import os.path
from typing import Final, NamedTuple
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import (
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
Read a file into memory. Raise MediaFileStorageError if we can't.