import re
from .base import BIT
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLIdentifierPreparer
from ... import util
def visit_mod_binary(self, binary, operator, **kw):
    return self.process(binary.left, **kw) + ' % ' + self.process(binary.right, **kw)