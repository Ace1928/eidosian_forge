import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
class ThreadedMySQLDBHandle(MySQLDBHandle):

    def __init__(self, fn, mode, max_age=None, bound=None):
        self.bound = bound
        if self.bound:
            self.db_queue = Queue.Queue()
        MySQLDBHandle.__init__(self, fn, mode, max_age=max_age)

    def _get_connection(self):
        if self.bound:
            return self.db_queue.get()
        else:
            return self._get_new_connection()

    def _release_connection(self, db):
        if self.bound:
            self.db_queue.put(db)
        else:
            db.close()

    def _safe_call(self, name, method, args):
        db = self._get_connection()
        try:
            return method(*args, db=db)
        except (MySQLdb.Error, AttributeError) as ex:
            self.log.error('%s failed: %s', name, ex)
            if not self.bound:
                raise DatabaseError('Database temporarily unavailable.')
            try:
                db.ping(True)
                return method(*args, db=db)
            except (MySQLdb.Error, AttributeError) as ex:
                db = self._reconnect(db)
                raise DatabaseError('Database temporarily unavailable.')
        finally:
            self._release_connection(db)

    def reconnect(self):
        if not self.bound:
            return
        for _ in xrange(self.bound):
            self.db_queue.put(self._get_new_connection())

    def _reconnect(self, db):
        if not self._check_reconnect_time():
            return db
        else:
            self.last_connect_attempt = time.time()
            return self._get_new_connection()

    def __del__(self):
        if not self.bound:
            return
        for db in iter(self.db_queue.get_nowait):
            try:
                db.close()
            except MySQLdb.Error:
                continue
            except Queue.Empty:
                break