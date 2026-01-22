import re
from .base import MSDialect
from .base import MSIdentifierPreparer
from ... import types as sqltypes
from ... import util
from ...engine import processors
class MSIdentifierPreparer_pymssql(MSIdentifierPreparer):

    def __init__(self, dialect):
        super().__init__(dialect)
        self._double_percents = False