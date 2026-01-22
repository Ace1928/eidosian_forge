import sys
from django.core.exceptions import ImproperlyConfigured
from django.db.backends.base.creation import BaseDatabaseCreation
from django.db.backends.postgresql.psycopg_any import errors
from django.db.backends.utils import strip_quotes
def _database_exists(self, cursor, database_name):
    cursor.execute('SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s', [strip_quotes(database_name)])
    return cursor.fetchone() is not None