import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
@observe('db')
def _db_changed(self, change):
    """validate the db, since it can be an Instance of two different types"""
    new = change['new']
    connection_types = (DummyDB, sqlite3.Connection)
    if not isinstance(new, connection_types):
        msg = '%s.db must be sqlite3 Connection or DummyDB, not %r' % (self.__class__.__name__, new)
        raise TraitError(msg)