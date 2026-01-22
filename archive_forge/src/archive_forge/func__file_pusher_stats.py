import concurrent.futures
import logging
import os
import queue
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
import wandb.util
from wandb.filesync import stats, step_checksum, step_upload
from wandb.sdk.lib.paths import LogicalPath
def _file_pusher_stats(self) -> None:
    while not self._stats_thread_stop.is_set():
        logger.info(f'FilePusher stats: {self._stats._stats}')
        time.sleep(1)