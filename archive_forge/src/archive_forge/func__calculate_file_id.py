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
def _calculate_file_id(data: bytes, mimetype: str, filename: str | None=None) -> str:
    """Hash data, mimetype, and an optional filename to generate a stable file ID.

    Parameters
    ----------
    data
        Content of in-memory file in bytes. Other types will throw TypeError.
    mimetype
        Any string. Will be converted to bytes and used to compute a hash.
    filename
        Any string. Will be converted to bytes and used to compute a hash.
    """
    filehash = hashlib.new('sha224', **HASHLIB_KWARGS)
    filehash.update(data)
    filehash.update(bytes(mimetype.encode()))
    if filename is not None:
        filehash.update(bytes(filename.encode()))
    return filehash.hexdigest()