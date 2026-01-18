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
def url_argument_spec():
    """
    Creates an argument spec that can be used with any module
    that will be requesting content via urllib/urllib2
    """
    return dict(url=dict(type='str'), force=dict(type='bool', default=False), http_agent=dict(type='str', default='ansible-httpget'), use_proxy=dict(type='bool', default=True), validate_certs=dict(type='bool', default=True), url_username=dict(type='str'), url_password=dict(type='str', no_log=True), force_basic_auth=dict(type='bool', default=False), client_cert=dict(type='path'), client_key=dict(type='path'), use_gssapi=dict(type='bool', default=False))