import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_unstartedClose(self):
    """
        If L{ConnectionPool.close} is called without L{ConnectionPool.start}
        having been called, the pool's startup event is cancelled.
        """
    reactor = EventReactor(False)
    pool = ConnectionPool('twisted.test.test_adbapi', cp_reactor=reactor)
    self.assertEqual(reactor.triggers, [('after', 'startup', pool._start)])
    pool.close()
    self.assertFalse(reactor.triggers)