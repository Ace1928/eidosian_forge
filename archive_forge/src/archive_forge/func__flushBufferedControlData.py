import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
def _flushBufferedControlData(self, *args):
    """
        Called when the connection is marked writable again after being marked unwritable.
        Attempts to flush buffered control data if there is any.
        """
    while self._consumerBlocked is None and self._bufferedControlFrames:
        nextWrite = self._bufferedControlFrames.popleft()
        self._bufferedControlFrameBytes -= len(nextWrite)
        self.transport.write(nextWrite)