import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
class OnlyProducerTransport:
    """
    Transport which isn't really a transport, just looks like one to
    someone not looking very hard.
    """
    paused = False
    disconnecting = False

    def __init__(self):
        self.data = []

    def pauseProducing(self):
        self.paused = True

    def resumeProducing(self):
        self.paused = False

    def write(self, bytes):
        self.data.append(bytes)