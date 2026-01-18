from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
@contextmanager
def unix_socket_patch_httpconnection_connect():
    """Monkey patch ``httplib.HTTPConnection.connect`` to be ``UnixHTTPConnection.connect``
        so that when calling ``super(UnixHTTPSConnection, self).connect()`` we get the
        correct behavior of creating self.sock for the unix socket
        """
    _connect = httplib.HTTPConnection.connect
    httplib.HTTPConnection.connect = UnixHTTPConnection.connect
    yield
    httplib.HTTPConnection.connect = _connect