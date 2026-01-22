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
def _sendPrioritisedData(self, *args):
    """
        The data sending loop. This function repeatedly calls itself, either
        from L{Deferred}s or from
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}

        This function sends data on streams according to the rules of HTTP/2
        priority. It ensures that the data from each stream is interleved
        according to the priority signalled by the client, making sure that the
        connection is used with maximal efficiency.

        This function will execute if data is available: if all data is
        exhausted, the function will place a deferred onto the L{H2Connection}
        object and wait until it is called to resume executing.
        """
    if not self._stillProducing:
        return
    stream = None
    while stream is None:
        try:
            stream = next(self.priority)
        except priority.DeadlockError:
            assert self._sendingDeferred is None
            self._sendingDeferred = Deferred()
            self._sendingDeferred.addCallback(self._sendPrioritisedData)
            return
    if self._consumerBlocked is not None:
        self._consumerBlocked.addCallback(self._sendPrioritisedData)
        return
    self.resetTimeout()
    remainingWindow = self.conn.local_flow_control_window(stream)
    frameData = self._outboundStreamQueues[stream].popleft()
    maxFrameSize = min(self.conn.max_outbound_frame_size, remainingWindow)
    if frameData is _END_STREAM_SENTINEL:
        self.conn.end_stream(stream)
        self.transport.write(self.conn.data_to_send())
        self._requestDone(stream)
    else:
        if len(frameData) > maxFrameSize:
            excessData = frameData[maxFrameSize:]
            frameData = frameData[:maxFrameSize]
            self._outboundStreamQueues[stream].appendleft(excessData)
        if frameData:
            self.conn.send_data(stream, frameData)
            self.transport.write(self.conn.data_to_send())
        if not self._outboundStreamQueues[stream]:
            self.priority.block(stream)
        if self.remainingOutboundWindow(stream) <= 0:
            self.streams[stream].flowControlBlocked()
    self._reactor.callLater(0, self._sendPrioritisedData)