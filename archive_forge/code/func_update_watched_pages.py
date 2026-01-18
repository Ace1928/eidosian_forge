from __future__ import annotations
import collections
import os
import sys
import types
from pathlib import Path
from typing import Callable, Final
from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import (
def update_watched_pages(self) -> None:
    old_watched_pages = self._watched_pages
    new_pages_paths: set[str] = set()
    for page_info in get_pages(self._main_script_path).values():
        new_pages_paths.add(page_info['script_path'])
        if page_info['script_path'] not in old_watched_pages:
            self._register_watcher(page_info['script_path'], module_name=None)
    for old_page_path in old_watched_pages:
        if old_page_path not in new_pages_paths:
            self._deregister_watcher(old_page_path)
    self._watched_pages = new_pages_paths