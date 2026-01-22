import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def add_progress(self, on_progress: Callable[[MailboxProgress], None]) -> None:
    self._on_progress = on_progress