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
class HTTPGSSAPIAuthHandler(BaseHandler):
    """ Handles Negotiate/Kerberos support through the gssapi library. """
    AUTH_HEADER_PATTERN = re.compile('(?:.*)\\s*(Negotiate|Kerberos)\\s*([^,]*),?', re.I)
    handler_order = 480

    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password
        self._context = None

    def get_auth_value(self, headers):
        auth_match = self.AUTH_HEADER_PATTERN.search(headers.get('www-authenticate', ''))
        if auth_match:
            return (auth_match.group(1), base64.b64decode(auth_match.group(2)))

    def http_error_401(self, req, fp, code, msg, headers):
        if self._context:
            return
        parsed = generic_urlparse(urlparse(req.get_full_url()))
        auth_header = self.get_auth_value(headers)
        if not auth_header:
            return
        auth_protocol, in_token = auth_header
        username = None
        if self.username:
            username = gssapi.Name(self.username, name_type=gssapi.NameType.user)
        if username and self.password:
            if not hasattr(gssapi.raw, 'acquire_cred_with_password'):
                raise NotImplementedError('Platform GSSAPI library does not support gss_acquire_cred_with_password, cannot acquire GSSAPI credential with explicit username and password.')
            b_password = to_bytes(self.password, errors='surrogate_or_strict')
            cred = gssapi.raw.acquire_cred_with_password(username, b_password, usage='initiate').creds
        else:
            cred = gssapi.Credentials(name=username, usage='initiate')
        cbt = None
        cert = getpeercert(fp, True)
        if cert and platform.system() != 'Darwin':
            cert_hash = get_channel_binding_cert_hash(cert)
            if cert_hash:
                cbt = gssapi.raw.ChannelBindings(application_data=b'tls-server-end-point:' + cert_hash)
        target = gssapi.Name('HTTP@%s' % parsed['hostname'], gssapi.NameType.hostbased_service)
        self._context = gssapi.SecurityContext(usage='initiate', name=target, creds=cred, channel_bindings=cbt)
        resp = None
        while not self._context.complete:
            out_token = self._context.step(in_token)
            if not out_token:
                break
            auth_header = '%s %s' % (auth_protocol, to_native(base64.b64encode(out_token)))
            req.add_unredirected_header('Authorization', auth_header)
            resp = self.parent.open(req)
            auth_header = self.get_auth_value(resp.headers)
            if not auth_header:
                break
            in_token = auth_header[1]
        return resp