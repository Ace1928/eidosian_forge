import atexit
import logging
import os
import time
from concurrent.futures import Future
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Union
from .hf_api import IGNORE_GIT_FOLDER_PATTERNS, CommitInfo, CommitOperationAdd, HfApi
from .utils import filter_repo_objects
def _run_scheduler(self) -> None:
    """Dumb thread waiting between each scheduled push to Hub."""
    while True:
        self.last_future = self.trigger()
        time.sleep(self.every * 60)
        if self.__stopped:
            break