import ipaddress
import itertools
import logging
from collections import deque
from ipaddress import IPv4Address, IPv6Address
from typing import Dict, List, Optional, Union
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
from h2.exceptions import FrameTooLargeError, H2Error
from twisted.internet.defer import Deferred
from twisted.internet.error import TimeoutError
from twisted.internet.interfaces import IHandshakeListener, IProtocolNegotiationFactory
from twisted.internet.protocol import Factory, Protocol, connectionDone
from twisted.internet.ssl import Certificate
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.client import URI
from zope.interface import implementer
from scrapy.core.http2.stream import Stream, StreamCloseReason
from scrapy.http import Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
def _send_pending_requests(self) -> None:
    """Initiate all pending requests from the deque following FIFO
        We make sure that at any time {allowed_max_concurrent_streams}
        streams are active.
        """
    while self._pending_request_stream_pool and self.metadata['active_streams'] < self.allowed_max_concurrent_streams and self.h2_connected:
        self.metadata['active_streams'] += 1
        stream = self._pending_request_stream_pool.popleft()
        stream.initiate_request()
        self._write_to_transport()