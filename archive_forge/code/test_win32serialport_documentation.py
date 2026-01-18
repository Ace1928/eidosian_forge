import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest

        Test the port is connected at initialization time, and
        C{Protocol.makeConnection} has been called on the desired protocol.
        