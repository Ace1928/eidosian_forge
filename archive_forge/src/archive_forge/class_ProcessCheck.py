import atexit
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
import psutil
import wandb
from ..interface.interface_queue import InterfaceQueue
from ..lib import tracelog
from . import context, handler, internal_util, sender, writer
class ProcessCheck:
    """Class to help watch a process id to detect when it is dead."""
    check_process_last: Optional[float]

    def __init__(self, settings: 'SettingsStatic', user_pid: Optional[int]) -> None:
        self.settings = settings
        self.pid = user_pid
        self.check_process_last = None
        self.check_process_interval = settings._internal_check_process

    def is_dead(self) -> bool:
        if not self.check_process_interval or not self.pid:
            return False
        time_now = time.time()
        if self.check_process_last and time_now < self.check_process_last + self.check_process_interval:
            return False
        self.check_process_last = time_now
        exists = psutil.pid_exists(self.pid)
        if not exists:
            logger.warning(f'Internal process exiting, parent pid {self.pid} disappeared')
            return True
        return False