import re
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .base import MySQLIdentifierPreparer
from .base import TEXT
from ... import sql
from ... import util
def _check_unicode_returns(self, connection):
    collation = connection.exec_driver_sql("show collation where %s = 'utf8mb4' and %s = 'utf8mb4_bin'" % (self.identifier_preparer.quote('Charset'), self.identifier_preparer.quote('Collation'))).scalar()
    has_utf8mb4_bin = self.server_version_info > (5,) and collation
    if has_utf8mb4_bin:
        additional_tests = [sql.collate(sql.cast(sql.literal_column("'test collated returns'"), TEXT(charset='utf8mb4')), 'utf8mb4_bin')]
    else:
        additional_tests = []
    return super()._check_unicode_returns(connection, additional_tests)