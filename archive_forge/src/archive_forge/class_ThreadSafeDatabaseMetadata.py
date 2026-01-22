import threading
from peewee import *
from peewee import Alias
from peewee import CompoundSelectQuery
from peewee import Metadata
from peewee import callable_
from peewee import __deprecated__
class ThreadSafeDatabaseMetadata(Metadata):
    """
    Metadata class to allow swapping database at run-time in a multi-threaded
    application. To use:

    class Base(Model):
        class Meta:
            model_metadata_class = ThreadSafeDatabaseMetadata
    """

    def __init__(self, *args, **kwargs):
        self._database = None
        self._local = threading.local()
        super(ThreadSafeDatabaseMetadata, self).__init__(*args, **kwargs)

    def _get_db(self):
        return getattr(self._local, 'database', self._database)

    def _set_db(self, db):
        if self._database is None:
            self._database = db
        self._local.database = db
    database = property(_get_db, _set_db)