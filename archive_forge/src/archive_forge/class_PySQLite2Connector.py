import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class PySQLite2Connector(DBTestConnector):
    """
    Connector that uses pysqlite's SQLite database support.
    """
    TEST_PREFIX = 'pysqlite2'
    escape_slashes = False
    num_iterations = 1

    def can_connect(self):
        if requireModule('pysqlite2.dbapi2') is None:
            return False
        else:
            return True

    def startDB(self):
        self.database = os.path.join(self.DB_DIR, self.DB_NAME)
        if os.path.exists(self.database):
            os.unlink(self.database)

    def getPoolArgs(self):
        args = ('pysqlite2.dbapi2',)
        kw = {'database': self.database, 'cp_max': 1, 'check_same_thread': False}
        return (args, kw)