from the connection pool, such as when using an ORM :class:`.Session` where
from working correctly.  The pysqlite DBAPI driver has several
import math
import os
import re
from .base import DATE
from .base import DATETIME
from .base import SQLiteDialect
from ... import exc
from ... import pool
from ... import types as sqltypes
from ... import util
@classmethod
def _is_url_file_db(cls, url):
    if (url.database and url.database != ':memory:') and url.query.get('mode', None) != 'memory':
        return True
    else:
        return False