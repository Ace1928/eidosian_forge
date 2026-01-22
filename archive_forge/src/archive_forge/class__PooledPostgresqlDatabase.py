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
class _PooledPostgresqlDatabase(PooledDatabase):

    def _is_closed(self, conn):
        if conn.closed:
            return True
        txn_status = conn.get_transaction_status()
        if txn_status == TRANSACTION_STATUS_UNKNOWN:
            return True
        elif txn_status != TRANSACTION_STATUS_IDLE:
            conn.rollback()
        return False

    def _can_reuse(self, conn):
        txn_status = conn.get_transaction_status()
        if txn_status == TRANSACTION_STATUS_UNKNOWN:
            return False
        elif txn_status == TRANSACTION_STATUS_INERROR:
            conn.reset()
        elif txn_status != TRANSACTION_STATUS_IDLE:
            conn.rollback()
        return True