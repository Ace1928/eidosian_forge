import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
class _BackgroundSyncer(Syncer):
    """Syncer using a background process for asynchronous file transfer."""

    def __init__(self, sync_period: float=DEFAULT_SYNC_PERIOD, sync_timeout: float=DEFAULT_SYNC_TIMEOUT):
        super(_BackgroundSyncer, self).__init__(sync_period=sync_period, sync_timeout=sync_timeout)
        self._sync_process = None
        self._current_cmd = None

    def _should_continue_existing_sync(self):
        """Returns whether a previous sync is still running within the timeout."""
        return self._sync_process and self._sync_process.is_running and (time.time() - self._sync_process.start_time < self.sync_timeout)

    def _launch_sync_process(self, sync_command: Tuple[Callable, Dict]):
        """Waits for the previous sync process to finish,
        then launches a new process that runs the given command."""
        if self._sync_process:
            try:
                self.wait()
            except Exception:
                logger.warning(f'Last sync command failed with the following error:\n{traceback.format_exc()}')
        self._current_cmd = sync_command
        self.retry()

    def sync_up(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
        if self._should_continue_existing_sync():
            logger.warning(f'Last sync still in progress, skipping sync up of {local_dir} to {remote_dir}')
            return False
        sync_up_cmd = self._sync_up_command(local_path=local_dir, uri=remote_dir, exclude=exclude)
        self._launch_sync_process(sync_up_cmd)
        return True

    def _sync_up_command(self, local_path: str, uri: str, exclude: Optional[List]=None) -> Tuple[Callable, Dict]:
        raise NotImplementedError

    def sync_down(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None) -> bool:
        if self._should_continue_existing_sync():
            logger.warning(f'Last sync still in progress, skipping sync down of {remote_dir} to {local_dir}')
            return False
        sync_down_cmd = self._sync_down_command(uri=remote_dir, local_path=local_dir)
        self._launch_sync_process(sync_down_cmd)
        return True

    def _sync_down_command(self, uri: str, local_path: str) -> Tuple[Callable, Dict]:
        raise NotImplementedError

    def delete(self, remote_dir: str) -> bool:
        if self._should_continue_existing_sync():
            logger.warning(f'Last sync still in progress, skipping deletion of {remote_dir}')
            return False
        delete_cmd = self._delete_command(uri=remote_dir)
        self._launch_sync_process(delete_cmd)
        return True

    def _delete_command(self, uri: str) -> Tuple[Callable, Dict]:
        raise NotImplementedError

    def wait(self):
        if self._sync_process:
            try:
                self._sync_process.wait(timeout=self.sync_timeout)
            except Exception as e:
                raise e
            finally:
                self._sync_process = None

    def retry(self):
        if not self._current_cmd:
            raise RuntimeError('No sync command set, cannot retry.')
        cmd, kwargs = self._current_cmd
        self._sync_process = _BackgroundProcess(cmd)
        self._sync_process.start(**kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_sync_process'] = None
        return state