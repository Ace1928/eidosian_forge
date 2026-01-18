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
def windowUpdated(self):
    """
        Called by the L{H2Connection} when this stream's flow control window
        has been opened.
        """
    if not self.producer:
        return
    if self._producerProducing:
        return
    remainingWindow = self._conn.remainingOutboundWindow(self.streamID)
    if not remainingWindow > 0:
        return
    self._producerProducing = True
    self.producer.resumeProducing()