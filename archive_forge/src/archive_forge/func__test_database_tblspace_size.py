import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _test_database_tblspace_size(self):
    return self._test_settings_get('DATAFILE_SIZE', default='50M')