import logging
import weakref
from threading import local as thread_local
from threading import Event
from threading import Thread
from peewee import __deprecated__
from playhouse.sqlite_ext import SqliteExtDatabase
class ThreadHelper(object):
    __slots__ = ('queue_max_size',)

    def __init__(self, queue_max_size=None):
        self.queue_max_size = queue_max_size

    def event(self):
        return Event()

    def queue(self, max_size=None):
        max_size = max_size if max_size is not None else self.queue_max_size
        return Queue(maxsize=max_size or 0)

    def thread(self, fn, *args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        return thread