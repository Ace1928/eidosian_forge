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
def _register_necessary_watchers(self, module_paths: dict[str, set[str]]) -> None:
    for name, paths in module_paths.items():
        for path in paths:
            if self._file_should_be_watched(path):
                self._register_watcher(str(Path(path).resolve()), name)