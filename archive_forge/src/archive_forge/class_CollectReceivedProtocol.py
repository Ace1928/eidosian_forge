import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
class CollectReceivedProtocol(Protocol):

    def __init__(self):
        self.received_data = []

    def dataReceived(self, data):
        self.received_data.append(data)