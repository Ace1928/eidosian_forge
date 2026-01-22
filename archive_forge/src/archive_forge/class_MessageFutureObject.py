import logging
import threading
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional
from ..lib import mailbox, tracelog
from .message_future import MessageFuture
class MessageFutureObject(MessageFuture):

    def __init__(self) -> None:
        super().__init__()

    def get(self, timeout: Optional[int]=None) -> Optional['pb.Result']:
        is_set = self._object_ready.wait(timeout)
        if is_set and self._object:
            return self._object
        return None