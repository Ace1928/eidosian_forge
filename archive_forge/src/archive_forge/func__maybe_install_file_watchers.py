from __future__ import annotations
import os
import threading
from copy import deepcopy
from typing import (
from blinker import Signal
import streamlit as st
import streamlit.watcher.path_watcher
from streamlit import file_util, runtime
from streamlit.logger import get_logger
def _maybe_install_file_watchers(self) -> None:
    with self._lock:
        if self._file_watchers_installed:
            return
        for path in self._file_paths:
            try:
                streamlit.watcher.path_watcher.watch_file(path, self._on_secrets_file_changed, watcher_type='poll')
            except FileNotFoundError:
                pass
        self._file_watchers_installed = True