import logging
import weakref
from threading import local as thread_local
from threading import Event
from threading import Thread
from peewee import __deprecated__
from playhouse.sqlite_ext import SqliteExtDatabase
def wait_unpause(self):
    obj = self.queue.get()
    if obj is UNPAUSE:
        logger.info('writer unpaused - reconnecting to database.')
        return True
    elif obj is SHUTDOWN:
        raise ShutdownException()
    elif obj is PAUSE:
        logger.error('writer received pause, but is already paused.')
    else:
        obj.set_result(None, WriterPaused())
        logger.warning('writer paused, not handling %s', obj)