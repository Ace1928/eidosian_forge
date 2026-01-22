import apsw
from peewee import *
from peewee import __exception_wrapper__
from peewee import BooleanField as _BooleanField
from peewee import DateField as _DateField
from peewee import DateTimeField as _DateTimeField
from peewee import DecimalField as _DecimalField
from peewee import Insert
from peewee import TimeField as _TimeField
from peewee import logger
from playhouse.sqlite_ext import SqliteExtDatabase
def _load_collations(self, conn):
    for name, fn in self._collations.items():
        conn.createcollation(name, fn)