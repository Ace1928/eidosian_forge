import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
class MailboxProbe:
    _result: Optional[pb.Result]
    _handle: Optional['MailboxHandle']

    def __init__(self) -> None:
        self._handle = None
        self._result = None

    def set_probe_result(self, result: pb.Result) -> None:
        self._result = result

    def get_probe_result(self) -> Optional[pb.Result]:
        return self._result

    def get_mailbox_handle(self) -> Optional['MailboxHandle']:
        return self._handle

    def set_mailbox_handle(self, handle: 'MailboxHandle') -> None:
        self._handle = handle