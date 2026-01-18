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
def update_watched_modules(self):
    if self._is_closed:
        return
    if set(sys.modules) != self._cached_sys_modules:
        modules_paths = {name: self._exclude_blacklisted_paths(get_module_paths(module)) for name, module in dict(sys.modules).items()}
        self._cached_sys_modules = set(sys.modules)
        self._register_necessary_watchers(modules_paths)