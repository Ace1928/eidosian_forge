import operator
from django.db import DataError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.utils.functional import cached_property
@cached_property
def uses_server_side_binding(self):
    options = self.connection.settings_dict['OPTIONS']
    return is_psycopg3 and options.get('server_side_binding') is True