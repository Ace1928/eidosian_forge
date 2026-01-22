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
def _call_htpasswd(args, stdin=None):
    """
    helper to run htpasswd cmd
    """
    if stdin is not None:
        stdin = stdin.encode('utf-8')
    proc = subprocess.Popen([htpasswd_path] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE if stdin else None)
    out, err = proc.communicate(stdin)
    rc = proc.wait()
    out = to_unicode_for_identify(out or '')
    return (out, rc)