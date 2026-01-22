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
def _load_aggregates(self, conn):
    for name, (klass, num_params) in self._aggregates.items():

        def make_aggregate():
            return (klass(), klass.step, klass.finalize)
        conn.createaggregatefunction(name, make_aggregate)