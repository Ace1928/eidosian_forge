import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _get_test_db_params(self):
    return {'dbname': self._test_database_name(), 'user': self._test_database_user(), 'password': self._test_database_passwd(), 'tblspace': self._test_database_tblspace(), 'tblspace_temp': self._test_database_tblspace_tmp(), 'datafile': self._test_database_tblspace_datafile(), 'datafile_tmp': self._test_database_tblspace_tmp_datafile(), 'maxsize': self._test_database_tblspace_maxsize(), 'maxsize_tmp': self._test_database_tblspace_tmp_maxsize(), 'size': self._test_database_tblspace_size(), 'size_tmp': self._test_database_tblspace_tmp_size(), 'extsize': self._test_database_tblspace_extsize(), 'extsize_tmp': self._test_database_tblspace_tmp_extsize()}