import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
class SimpleProtocol(protocol.ProcessProtocol):
    """
            A protocol that fires deferreds when connected and disconnected.
            """

    def makeConnection(self, transport):
        connected.callback(transport)

    def processEnded(self, reason):
        ended.callback(None)