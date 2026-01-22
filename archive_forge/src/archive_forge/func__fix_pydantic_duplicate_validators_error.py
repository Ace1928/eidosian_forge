from __future__ import annotations
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any, Final
from streamlit import (
from streamlit.config import CONFIG_FILENAMES
from streamlit.git_util import MIN_GIT_VERSION, GitRepo
from streamlit.logger import get_logger
from streamlit.source_util import invalidate_pages_cache
from streamlit.watcher import report_watchdog_availability, watch_dir, watch_file
from streamlit.web.server import Server, server_address_is_unix_socket, server_util
def _fix_pydantic_duplicate_validators_error():
    """Pydantic by default disallows to reuse of validators with the same name,
    this combined with the Streamlit execution model leads to an error on the second
    Streamlit script rerun if the Pydantic validator is registered
    in the streamlit script.

    It is important to note that the same issue exists for Pydantic validators inside
    Jupyter notebooks, https://github.com/pydantic/pydantic/issues/312 and in order
    to fix that in Pydantic they use the `in_ipython` function that checks that
    Pydantic runs not in `ipython` environment.

    Inside this function we patch `in_ipython` function to always return `True`.

    This change will relax rules for writing Pydantic validators inside
    Streamlit script a little bit, similar to how it works in jupyter,
    which should not be critical.
    """
    try:
        from pydantic import class_validators
        class_validators.in_ipython = lambda: True
    except ImportError:
        pass