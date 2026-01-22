import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class ConnectionCursorRaise:
    count = 0

    def reconnect(self):
        pass

    def cursor(self):
        if self.count == 0:
            self.count += 1
            raise RuntimeError('problem!')