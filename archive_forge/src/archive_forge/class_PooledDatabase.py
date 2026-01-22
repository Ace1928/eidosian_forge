import functools
import heapq
import logging
import random
import threading
import time
from collections import namedtuple
from itertools import chain
from peewee import MySQLDatabase
from peewee import PostgresqlDatabase
from peewee import SqliteDatabase
class PooledDatabase(object):

    def __init__(self, database, max_connections=20, stale_timeout=None, timeout=None, **kwargs):
        self._max_connections = make_int(max_connections)
        self._stale_timeout = make_int(stale_timeout)
        self._wait_timeout = make_int(timeout)
        if self._wait_timeout == 0:
            self._wait_timeout = float('inf')
        self._pool_lock = threading.RLock()
        self._connections = []
        self._in_use = {}
        self.conn_key = id
        super(PooledDatabase, self).__init__(database, **kwargs)

    def init(self, database, max_connections=None, stale_timeout=None, timeout=None, **connect_kwargs):
        super(PooledDatabase, self).init(database, **connect_kwargs)
        if max_connections is not None:
            self._max_connections = make_int(max_connections)
        if stale_timeout is not None:
            self._stale_timeout = make_int(stale_timeout)
        if timeout is not None:
            self._wait_timeout = make_int(timeout)
            if self._wait_timeout == 0:
                self._wait_timeout = float('inf')

    def connect(self, reuse_if_open=False):
        if not self._wait_timeout:
            return super(PooledDatabase, self).connect(reuse_if_open)
        expires = time.time() + self._wait_timeout
        while expires > time.time():
            try:
                ret = super(PooledDatabase, self).connect(reuse_if_open)
            except MaxConnectionsExceeded:
                time.sleep(0.1)
            else:
                return ret
        raise MaxConnectionsExceeded('Max connections exceeded, timed out attempting to connect.')

    @locked
    def _connect(self):
        while True:
            try:
                ts, conn = heapq.heappop(self._connections)
                key = self.conn_key(conn)
            except IndexError:
                ts = conn = None
                logger.debug('No connection available in pool.')
                break
            else:
                if self._is_closed(conn):
                    logger.debug('Connection %s was closed.', key)
                    ts = conn = None
                elif self._stale_timeout and self._is_stale(ts):
                    logger.debug('Connection %s was stale, closing.', key)
                    self._close(conn, True)
                    ts = conn = None
                else:
                    break
        if conn is None:
            if self._max_connections and len(self._in_use) >= self._max_connections:
                raise MaxConnectionsExceeded('Exceeded maximum connections.')
            conn = super(PooledDatabase, self)._connect()
            ts = time.time()
            key = self.conn_key(conn)
            logger.debug('Created new connection %s.', key)
        self._in_use[key] = PoolConnection(ts, conn, time.time())
        return conn

    def _is_stale(self, timestamp):
        return time.time() - timestamp > self._stale_timeout

    def _is_closed(self, conn):
        return False

    def _can_reuse(self, conn):
        return True

    @locked
    def _close(self, conn, close_conn=False):
        key = self.conn_key(conn)
        if close_conn:
            super(PooledDatabase, self)._close(conn)
        elif key in self._in_use:
            pool_conn = self._in_use.pop(key)
            if self._stale_timeout and self._is_stale(pool_conn.timestamp):
                logger.debug('Closing stale connection %s.', key)
                super(PooledDatabase, self)._close(conn)
            elif self._can_reuse(conn):
                logger.debug('Returning %s to pool.', key)
                heapq.heappush(self._connections, (pool_conn.timestamp, conn))
            else:
                logger.debug('Closed %s.', key)

    @locked
    def manual_close(self):
        """
        Close the underlying connection without returning it to the pool.
        """
        if self.is_closed():
            return False
        conn = self.connection()
        self._in_use.pop(self.conn_key(conn), None)
        self.close()
        self._close(conn, close_conn=True)

    @locked
    def close_idle(self):
        for _, conn in self._connections:
            self._close(conn, close_conn=True)
        self._connections = []

    @locked
    def close_stale(self, age=600):
        in_use = {}
        cutoff = time.time() - age
        n = 0
        for key, pool_conn in self._in_use.items():
            if pool_conn.checked_out < cutoff:
                self._close(pool_conn.connection, close_conn=True)
                n += 1
            else:
                in_use[key] = pool_conn
        self._in_use = in_use
        return n

    @locked
    def close_all(self):
        self.close()
        for _, conn in self._connections:
            self._close(conn, close_conn=True)
        for pool_conn in self._in_use.values():
            self._close(pool_conn.connection, close_conn=True)
        self._connections = []
        self._in_use = {}