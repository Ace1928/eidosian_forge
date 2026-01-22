import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
class PolicyLive(FileEventHandler):
    """Event handler that uploads respecting throttling.

    Uploads files every RATE_LIMIT_SECONDS, which changes as the size increases to deal
    with throttling.
    """
    RATE_LIMIT_SECONDS = 15
    unit_dict = dict(util.POW_10_BYTES)
    RATE_LIMIT_SIZE_INCREASE = 1.2

    def __init__(self, file_path: PathStr, save_name: LogicalPath, file_pusher: 'FilePusher', settings: Optional['SettingsStatic']=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(file_path, save_name, file_pusher, *args, **kwargs)
        self._last_uploaded_time: Optional[float] = None
        self._last_uploaded_size: int = 0
        if settings is not None:
            if settings._live_policy_rate_limit is not None:
                self.RATE_LIMIT_SECONDS = settings._live_policy_rate_limit
            self._min_wait_time: Optional[float] = settings._live_policy_wait_time
        else:
            self._min_wait_time = None

    @property
    def current_size(self) -> int:
        return os.path.getsize(self.file_path)

    @classmethod
    def min_wait_for_size(cls, size: int) -> float:
        if size < 10 * cls.unit_dict['MB']:
            return 60
        elif size < 100 * cls.unit_dict['MB']:
            return 5 * 60
        elif size < cls.unit_dict['GB']:
            return 10 * 60
        else:
            return 20 * 60

    def should_update(self) -> bool:
        if self._last_uploaded_time is not None:
            time_elapsed = time.time() - self._last_uploaded_time
            if time_elapsed < self.RATE_LIMIT_SECONDS:
                return False
            if float(self._last_uploaded_size) > 0:
                size_increase = self.current_size / float(self._last_uploaded_size)
                if size_increase < self.RATE_LIMIT_SIZE_INCREASE:
                    return False
            return time_elapsed > (self._min_wait_time or self.min_wait_for_size(self.current_size))
        return True

    def on_modified(self, force: bool=False) -> None:
        if self.current_size == 0:
            return
        if self._last_sync == os.path.getmtime(self.file_path):
            return
        if force or self.should_update():
            self.save_file()

    def save_file(self) -> None:
        self._last_sync = os.path.getmtime(self.file_path)
        self._last_uploaded_time = time.time()
        self._last_uploaded_size = self.current_size
        self._file_pusher.file_changed(self.save_name, self.file_path)

    def finish(self) -> None:
        self.on_modified(force=True)

    @property
    def policy(self) -> 'PolicyName':
        return 'live'