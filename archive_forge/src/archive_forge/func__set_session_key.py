import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def _set_session_key(self, value):
    """
        Validate session key on assignment. Invalid values will set to None.
        """
    if self._validate_session_key(value):
        self.__session_key = value
    else:
        self.__session_key = None