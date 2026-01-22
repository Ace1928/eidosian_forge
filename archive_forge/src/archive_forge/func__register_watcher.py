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
def _register_watcher(self, filepath, module_name):
    global PathWatcher
    if PathWatcher is None:
        PathWatcher = get_default_path_watcher_class()
    if PathWatcher is NoOpPathWatcher:
        return
    try:
        wm = WatchedModule(watcher=PathWatcher(filepath, self.on_file_changed), module_name=module_name)
    except PermissionError:
        return
    self._watched_modules[filepath] = wm