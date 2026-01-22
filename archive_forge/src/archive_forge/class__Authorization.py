import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
class _Authorization(_SingleValueHeader):
    """
    Authorization, RFC 2617 (RFC 2616, 14.8)
    """

    def compose(self, digest=None, basic=None, username=None, password=None, challenge=None, path=None, method=None):
        assert username and password
        if basic or not challenge:
            assert not digest
            userpass = '%s:%s' % (username.strip(), password.strip())
            return 'Basic %s' % userpass.encode('base64').strip()
        assert challenge and (not basic)
        path = path or '/'
        _, realm = challenge.split('realm="')
        realm, _ = realm.split('"', 1)
        auth = AbstractDigestAuthHandler()
        auth.add_password(realm, path, username, password)
        token, challenge = challenge.split(' ', 1)
        chal = parse_keqv_list(parse_http_list(challenge))

        class FakeRequest(object):

            @property
            def full_url(self):
                return path
            selector = full_url

            @property
            def data(self):
                return None

            def get_method(self):
                return method or 'GET'
        retval = 'Digest %s' % auth.get_authorization(FakeRequest(), chal)
        return (retval,)