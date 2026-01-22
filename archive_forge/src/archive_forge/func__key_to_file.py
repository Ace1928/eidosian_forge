import datetime
import logging
import os
import shutil
import tempfile
from django.conf import settings
from django.contrib.sessions.backends.base import (
from django.contrib.sessions.exceptions import InvalidSessionKey
from django.core.exceptions import ImproperlyConfigured, SuspiciousOperation
def _key_to_file(self, session_key=None):
    """
        Get the file associated with this session key.
        """
    if session_key is None:
        session_key = self._get_or_create_session_key()
    if not set(session_key).issubset(VALID_KEY_CHARS):
        raise InvalidSessionKey('Invalid characters in session key')
    return os.path.join(self.storage_path, self.file_prefix + session_key)