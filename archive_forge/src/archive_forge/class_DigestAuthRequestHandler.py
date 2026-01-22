import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class DigestAuthRequestHandler(AuthRequestHandler):
    """Implements the digest authentication of a request.

    We need persistence for some attributes and that can't be
    achieved here since we get instantiated for each request. We
    rely on the DigestAuthServer to take care of them.
    """

    def authorized(self):
        tcs = self.server.test_case_server
        auth_header = self.headers.get(tcs.auth_header_recv, None)
        if auth_header is None:
            return False
        scheme, auth = auth_header.split(None, 1)
        if scheme.lower() == tcs.auth_scheme:
            auth_dict = parse_keqv_list(parse_http_list(auth))
            return tcs.digest_authorized(auth_dict, self.command)
        return False

    def send_header_auth_reqed(self):
        tcs = self.server.test_case_server
        header = 'Digest realm="%s", ' % tcs.auth_realm
        header += 'nonce="{}", algorithm="{}", qop="auth"'.format(tcs.auth_nonce, 'MD5')
        self.send_header(tcs.auth_header_sent, header)