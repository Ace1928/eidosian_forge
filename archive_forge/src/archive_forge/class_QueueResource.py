import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
class QueueResource(Resource):
    """
    Add all requests to an internal queue,
    without responding to the requests.
    You can access the requests from the queue and handle their response.
    """
    isLeaf = True

    def __init__(self) -> None:
        super().__init__()
        self.dispatchedRequests: List[Request] = []

    def render_GET(self, request: Request) -> int:
        self.dispatchedRequests.append(request)
        return NOT_DONE_YET