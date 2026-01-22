import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class DummyConnectionPool(ConnectionPool):
    """
    A testable L{ConnectionPool};
    """
    threadpool = NonThreadPool()

    def __init__(self):
        """
        Don't forward init call.
        """
        self._reactor = reactor