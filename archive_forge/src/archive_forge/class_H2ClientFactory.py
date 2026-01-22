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
@implementer(IProtocolNegotiationFactory)
class H2ClientFactory(Factory):

    def __init__(self, uri: URI, settings: Settings, conn_lost_deferred: Deferred) -> None:
        self.uri = uri
        self.settings = settings
        self.conn_lost_deferred = conn_lost_deferred

    def buildProtocol(self, addr) -> H2ClientProtocol:
        return H2ClientProtocol(self.uri, self.settings, self.conn_lost_deferred)

    def acceptableProtocols(self) -> List[bytes]:
        return [PROTOCOL_NAME]