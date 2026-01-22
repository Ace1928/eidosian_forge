from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
def _set_record(self, key, value):
    """
        helper for setting record which takes care of inserting source line if needed;

        :returns:
            bool if key already present
        """
    records = self._records
    existing = key in records
    records[key] = value
    if not existing:
        self._source.append((_RECORD, key))
    return existing