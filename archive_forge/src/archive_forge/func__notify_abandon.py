import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _notify_abandon(self) -> None:
    self._abandoned = True
    with self._lock:
        self._event.set()
    if self._wait_all:
        self._wait_all.notify()