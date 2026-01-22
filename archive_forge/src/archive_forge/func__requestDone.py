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
def _requestDone(self, streamID):
    """
        Called internally by the data sending loop to clean up state that was
        being used for the stream. Called when the stream is complete.

        @param streamID: The ID of the stream to clean up state for.
        @type streamID: L{int}
        """
    del self._outboundStreamQueues[streamID]
    self.priority.remove_stream(streamID)
    del self.streams[streamID]
    cleanupCallback = self._streamCleanupCallbacks.pop(streamID)
    cleanupCallback.callback(streamID)