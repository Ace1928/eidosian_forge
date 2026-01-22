import re
from .base import BIT
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLIdentifierPreparer
from ... import util
def _compat_fetchone(self, rp, charset=None):
    return rp.fetchone()