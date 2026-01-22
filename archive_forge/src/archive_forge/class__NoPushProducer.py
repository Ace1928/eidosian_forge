from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
@implementer(interfaces.IPushProducer)
class _NoPushProducer:
    """
    A no-op version of L{interfaces.IPushProducer}, used to abstract over the
    possibility that a L{HTTPChannel} transport does not provide
    L{IPushProducer}.
    """

    def pauseProducing(self):
        """
        Pause producing data.

        Tells a producer that it has produced too much data to process for
        the time being, and to stop until resumeProducing() is called.
        """

    def resumeProducing(self):
        """
        Resume producing data.

        This tells a producer to re-add itself to the main loop and produce
        more data for its consumer.
        """

    def registerProducer(self, producer, streaming):
        """
        Register to receive data from a producer.

        @param producer: The producer to register.
        @param streaming: Whether this is a streaming producer or not.
        """

    def unregisterProducer(self):
        """
        Stop consuming data from a producer, without disconnecting.
        """

    def stopProducing(self):
        """
        IProducer.stopProducing
        """