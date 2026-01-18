import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_startedClose(self):
    """
        If L{ConnectionPool.close} is called after it has been started, but
        not by its shutdown trigger, the shutdown trigger is cancelled.
        """
    reactor = EventReactor(True)
    pool = ConnectionPool('twisted.test.test_adbapi', cp_reactor=reactor)
    self.assertEqual(reactor.triggers, [('during', 'shutdown', pool.finalClose)])
    pool.close()
    self.assertFalse(reactor.triggers)