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
def _streamIsActive(self, streamID):
    """
        Checks whether Twisted has still got state for a given stream and so
        can process events for that stream.

        @param streamID: The ID of the stream that needs processing.
        @type streamID: L{int}

        @return: Whether the stream still has state allocated.
        @rtype: L{bool}
        """
    return streamID in self.streams