import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _deliver(self, result: pb.Result) -> None:
    with self._lock:
        self._result = result
        self._event.set()
    if self._wait_all:
        self._wait_all.notify()