from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
def _get_or_identify_record(self, hash, scheme=None, category=None):
    """return record based on scheme, or failing that, by identifying hash"""
    if scheme:
        if not isinstance(hash, unicode_or_bytes_types):
            raise ExpectedStringError(hash, 'hash')
        return self._get_record(scheme, category)
    else:
        return self._identify_record(hash, category)