import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _add_handle(self, handle: 'MailboxHandle') -> None:
    handle._slot._set_wait_all(self)
    self._handles.append(handle)
    if handle._slot._event.is_set():
        self._event.set()