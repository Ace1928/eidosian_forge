import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
@cached_property
def _maindb_connection(self):
    """
        This is analogous to other backends' `_nodb_connection` property,
        which allows access to an "administrative" connection which can
        be used to manage the test databases.
        For Oracle, the only connection that can be used for that purpose
        is the main (non-test) connection.
        """
    settings_dict = settings.DATABASES[self.connection.alias]
    user = settings_dict.get('SAVED_USER') or settings_dict['USER']
    password = settings_dict.get('SAVED_PASSWORD') or settings_dict['PASSWORD']
    settings_dict = {**settings_dict, 'USER': user, 'PASSWORD': password}
    DatabaseWrapper = type(self.connection)
    return DatabaseWrapper(settings_dict, alias=self.connection.alias)