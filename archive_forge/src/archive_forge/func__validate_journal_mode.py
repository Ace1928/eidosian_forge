import logging
import weakref
from threading import local as thread_local
from threading import Event
from threading import Thread
from peewee import __deprecated__
from playhouse.sqlite_ext import SqliteExtDatabase
def _validate_journal_mode(self, pragmas=None):
    if not pragmas:
        return {'journal_mode': 'wal'}
    if not isinstance(pragmas, dict):
        pragmas = dict(((k.lower(), v) for k, v in pragmas))
    if pragmas.get('journal_mode', 'wal').lower() != 'wal':
        raise ValueError(self.WAL_MODE_ERROR_MESSAGE)
    pragmas['journal_mode'] = 'wal'
    return pragmas