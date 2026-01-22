import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class FirebirdConnector(DBTestConnector):
    TEST_PREFIX = 'Firebird'
    test_failures = False
    escape_slashes = False
    good_sql = None
    can_clear = False
    num_iterations = 5

    def can_connect(self):
        if requireModule('kinterbasdb') is None:
            return False
        try:
            self.startDB()
            self.stopDB()
            return True
        except BaseException:
            return False

    def startDB(self):
        import kinterbasdb
        self.DB_NAME = os.path.join(self.DB_DIR, DBTestConnector.DB_NAME)
        os.chmod(self.DB_DIR, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
        sql = 'create database "%s" user "%s" password "%s"'
        sql %= (self.DB_NAME, self.DB_USER, self.DB_PASS)
        conn = kinterbasdb.create_database(sql)
        conn.close()

    def getPoolArgs(self):
        args = ('kinterbasdb',)
        kw = {'database': self.DB_NAME, 'host': '127.0.0.1', 'user': self.DB_USER, 'password': self.DB_PASS}
        return (args, kw)

    def stopDB(self):
        import kinterbasdb
        conn = kinterbasdb.connect(database=self.DB_NAME, host='127.0.0.1', user=self.DB_USER, password=self.DB_PASS)
        conn.drop_database()