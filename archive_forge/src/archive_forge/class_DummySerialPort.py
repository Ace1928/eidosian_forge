from twisted.internet.error import ConnectionDone
from twisted.internet.protocol import Protocol
from twisted.python.failure import Failure
from twisted.trial import unittest
class DummySerialPort(serialport.SerialPort):
    _serialFactory = DoNothing

    def _finishPortSetup(self):
        pass