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
def RedirectHandlerFactory(follow_redirects=None, validate_certs=True, ca_path=None, ciphers=None):
    """This is a class factory that closes over the value of
    ``follow_redirects`` so that the RedirectHandler class has access to
    that value without having to use globals, and potentially cause problems
    where ``open_url`` or ``fetch_url`` are used multiple times in a module.
    """

    class RedirectHandler(urllib_request.HTTPRedirectHandler):
        """This is an implementation of a RedirectHandler to match the
        functionality provided by httplib2. It will utilize the value of
        ``follow_redirects`` that is passed into ``RedirectHandlerFactory``
        to determine how redirects should be handled in urllib2.
        """

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if not any((HAS_SSLCONTEXT, HAS_URLLIB3_PYOPENSSLCONTEXT)):
                handler = maybe_add_ssl_handler(newurl, validate_certs, ca_path=ca_path, ciphers=ciphers)
                if handler:
                    urllib_request._opener.add_handler(handler)
            if follow_redirects == 'urllib2':
                return urllib_request.HTTPRedirectHandler.redirect_request(self, req, fp, code, msg, headers, newurl)
            elif follow_redirects in ['no', 'none', False]:
                raise urllib_error.HTTPError(newurl, code, msg, headers, fp)
            method = req.get_method()
            if follow_redirects in ['all', 'yes', True]:
                if code < 300 or code >= 400:
                    raise urllib_error.HTTPError(req.get_full_url(), code, msg, headers, fp)
            elif follow_redirects == 'safe':
                if code < 300 or code >= 400 or method not in ('GET', 'HEAD'):
                    raise urllib_error.HTTPError(req.get_full_url(), code, msg, headers, fp)
            else:
                raise urllib_error.HTTPError(req.get_full_url(), code, msg, headers, fp)
            try:
                data = req.get_data()
                origin_req_host = req.get_origin_req_host()
            except AttributeError:
                data = req.data
                origin_req_host = req.origin_req_host
            newurl = newurl.replace(' ', '%20')
            if code in (307, 308):
                req_headers = req.headers
            else:
                data = None
                req_headers = dict(((k, v) for k, v in req.headers.items() if k.lower() not in ('content-length', 'content-type', 'transfer-encoding')))
                if code == 303 and method != 'HEAD':
                    method = 'GET'
                if code == 302 and method != 'HEAD':
                    method = 'GET'
                if code == 301 and method == 'POST':
                    method = 'GET'
            return RequestWithMethod(newurl, method=method, headers=req_headers, data=data, origin_req_host=origin_req_host, unverifiable=True)
    return RedirectHandler