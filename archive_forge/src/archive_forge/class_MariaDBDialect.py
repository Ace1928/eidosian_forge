from .base import MariaDBIdentifierPreparer
from .base import MySQLDialect
class MariaDBDialect(MySQLDialect):
    is_mariadb = True
    supports_statement_cache = True
    name = 'mariadb'
    preparer = MariaDBIdentifierPreparer