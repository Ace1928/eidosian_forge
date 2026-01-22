import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def _get_test_db_name(self):
    """
        Internal implementation - return the name of the test DB that will be
        created. Only useful when called from create_test_db() and
        _create_test_db() and when no external munging is done with the 'NAME'
        settings.
        """
    if self.connection.settings_dict['TEST']['NAME']:
        return self.connection.settings_dict['TEST']['NAME']
    return TEST_DATABASE_PREFIX + self.connection.settings_dict['NAME']