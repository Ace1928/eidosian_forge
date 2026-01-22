from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Final
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
def _check_if_path_changed(self) -> None:
    if not self._active:
        return
    modification_time = util.path_modification_time(self._path, self._allow_nonexistent)
    if modification_time != 0.0 and modification_time <= self._modification_time:
        self._schedule()
        return
    self._modification_time = modification_time
    md5 = util.calc_md5_with_blocking_retries(self._path, glob_pattern=self._glob_pattern, allow_nonexistent=self._allow_nonexistent)
    if md5 == self._md5:
        self._schedule()
        return
    self._md5 = md5
    _LOGGER.debug('Change detected: %s', self._path)
    self._on_changed(self._path)
    self._schedule()