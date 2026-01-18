import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_rollbackErrorLogged(self):
    """
        If an error happens during rollback, L{ConnectionLost} is raised but
        the original error is logged.
        """

    class ConnectionRollbackRaise:

        def rollback(self):
            raise RuntimeError('problem!')
    pool = FakePool(ConnectionRollbackRaise)
    connection = Connection(pool)
    self.assertRaises(ConnectionLost, connection.rollback)
    errors = self.flushLoggedErrors(RuntimeError)
    self.assertEqual(len(errors), 1)
    self.assertEqual(errors[0].value.args[0], 'problem!')