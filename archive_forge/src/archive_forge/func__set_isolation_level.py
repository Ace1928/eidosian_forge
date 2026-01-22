import re
from .base import BIT
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLIdentifierPreparer
from ... import util
def _set_isolation_level(self, connection, level):
    if level == 'AUTOCOMMIT':
        connection.autocommit = True
    else:
        connection.autocommit = False
        super()._set_isolation_level(connection, level)