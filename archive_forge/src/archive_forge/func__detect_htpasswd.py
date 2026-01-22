from __future__ import with_statement
from logging import getLogger
import os
import subprocess
from passlib import apache, registry
from passlib.exc import MissingBackendError
from passlib.utils.compat import irange
from passlib.tests.backports import unittest
from passlib.tests.utils import TestCase, get_file, set_file, ensure_mtime_changed
from passlib.utils.compat import u
from passlib.utils import to_bytes
from passlib.utils.handlers import to_unicode_for_identify
def _detect_htpasswd():
    """
    helper to check if htpasswd is present
    """
    try:
        out, rc = _call_htpasswd([])
    except OSError:
        return (False, False)
    if not rc:
        log.warning('htpasswd test returned with rc=0')
    have_bcrypt = ' -B ' in out
    return (True, have_bcrypt)