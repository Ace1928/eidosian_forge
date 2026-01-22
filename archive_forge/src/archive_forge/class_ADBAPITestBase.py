import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class ADBAPITestBase:
    """
    Test the asynchronous DB-API code.
    """
    openfun_called: Dict[object, bool] = {}
    if interfaces.IReactorThreads(reactor, None) is None:
        skip = 'ADB-API requires threads, no way to test without them'

    def extraSetUp(self):
        """
        Set up the database and create a connection pool pointing at it.
        """
        self.startDB()
        self.dbpool = self.makePool(cp_openfun=self.openfun)
        self.dbpool.start()

    def tearDown(self):
        d = self.dbpool.runOperation('DROP TABLE simple')
        d.addCallback(lambda res: self.dbpool.close())
        d.addCallback(lambda res: self.stopDB())
        return d

    def openfun(self, conn):
        self.openfun_called[conn] = True

    def checkOpenfunCalled(self, conn=None):
        if not conn:
            self.assertTrue(self.openfun_called)
        else:
            self.assertIn(conn, self.openfun_called)

    def test_pool(self):
        d = self.dbpool.runOperation(simple_table_schema)
        if self.test_failures:
            d.addCallback(self._testPool_1_1)
            d.addCallback(self._testPool_1_2)
            d.addCallback(self._testPool_1_3)
            d.addCallback(self._testPool_1_4)
            d.addCallback(lambda res: self.flushLoggedErrors())
        d.addCallback(self._testPool_2)
        d.addCallback(self._testPool_3)
        d.addCallback(self._testPool_4)
        d.addCallback(self._testPool_5)
        d.addCallback(self._testPool_6)
        d.addCallback(self._testPool_7)
        d.addCallback(self._testPool_8)
        d.addCallback(self._testPool_9)
        return d

    def _testPool_1_1(self, res):
        d = defer.maybeDeferred(self.dbpool.runQuery, 'select * from NOTABLE')
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_1_2(self, res):
        d = defer.maybeDeferred(self.dbpool.runOperation, 'deletexxx from NOTABLE')
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_1_3(self, res):
        d = defer.maybeDeferred(self.dbpool.runInteraction, self.bad_interaction)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_1_4(self, res):
        d = defer.maybeDeferred(self.dbpool.runWithConnection, self.bad_withConnection)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_2(self, res):
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            self.assertTrue(int(row[0][0]) == 0, 'Interaction not rolled back')
            self.checkOpenfunCalled()
        d.addCallback(_check)
        return d

    def _testPool_3(self, res):
        sql = 'select count(1) from simple'
        inserts = []
        for i in range(self.num_iterations):
            sql = 'insert into simple(x) values(%d)' % i
            inserts.append(self.dbpool.runOperation(sql))
        d = defer.gatherResults(inserts)

        def _select(res):
            sql = 'select x from simple order by x'
            d = self.dbpool.runQuery(sql)
            return d
        d.addCallback(_select)

        def _check(rows):
            self.assertTrue(len(rows) == self.num_iterations, 'Wrong number of rows')
            for i in range(self.num_iterations):
                self.assertTrue(len(rows[i]) == 1, 'Wrong size row')
                self.assertTrue(rows[i][0] == i, 'Values not returned.')
        d.addCallback(_check)
        return d

    def _testPool_4(self, res):
        d = self.dbpool.runInteraction(self.interaction)
        d.addCallback(lambda res: self.assertEqual(res, 'done'))
        return d

    def _testPool_5(self, res):
        d = self.dbpool.runWithConnection(self.withConnection)
        d.addCallback(lambda res: self.assertEqual(res, 'done'))
        return d

    def _testPool_6(self, res):
        d = self.dbpool.runWithConnection(self.close_withConnection)
        return d

    def _testPool_7(self, res):
        ds = []
        for i in range(self.num_iterations):
            sql = 'select x from simple where x = %d' % i
            ds.append(self.dbpool.runQuery(sql))
        dlist = defer.DeferredList(ds, fireOnOneErrback=True)

        def _check(result):
            for i in range(self.num_iterations):
                self.assertTrue(result[i][1][0][0] == i, 'Value not returned')
        dlist.addCallback(_check)
        return dlist

    def _testPool_8(self, res):
        ds = []
        for i in range(self.num_iterations):
            sql = 'delete from simple where x = %d' % i
            ds.append(self.dbpool.runOperation(sql))
        dlist = defer.DeferredList(ds, fireOnOneErrback=True)
        return dlist

    def _testPool_9(self, res):
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            self.assertTrue(int(row[0][0]) == 0, "Didn't successfully delete table contents")
            self.checkConnect()
        d.addCallback(_check)
        return d

    def checkConnect(self):
        """Check the connect/disconnect synchronous calls."""
        conn = self.dbpool.connect()
        self.checkOpenfunCalled(conn)
        curs = conn.cursor()
        curs.execute('insert into simple(x) values(1)')
        curs.execute('select x from simple')
        res = curs.fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(res[0][0], 1)
        curs.execute('delete from simple')
        curs.execute('select x from simple')
        self.assertEqual(len(curs.fetchall()), 0)
        curs.close()
        self.dbpool.disconnect(conn)

    def interaction(self, transaction):
        transaction.execute('select x from simple order by x')
        for i in range(self.num_iterations):
            row = transaction.fetchone()
            self.assertTrue(len(row) == 1, 'Wrong size row')
            self.assertTrue(row[0] == i, 'Value not returned.')
        self.assertIsNone(transaction.fetchone(), 'Too many rows')
        return 'done'

    def bad_interaction(self, transaction):
        if self.can_rollback:
            transaction.execute('insert into simple(x) values(0)')
        transaction.execute('select * from NOTABLE')

    def withConnection(self, conn):
        curs = conn.cursor()
        try:
            curs.execute('select x from simple order by x')
            for i in range(self.num_iterations):
                row = curs.fetchone()
                self.assertTrue(len(row) == 1, 'Wrong size row')
                self.assertTrue(row[0] == i, 'Value not returned.')
        finally:
            curs.close()
        return 'done'

    def close_withConnection(self, conn):
        conn.close()

    def bad_withConnection(self, conn):
        curs = conn.cursor()
        try:
            curs.execute('select * from NOTABLE')
        finally:
            curs.close()