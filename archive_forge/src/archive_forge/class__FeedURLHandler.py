import base64
import datetime
import gzip
import io
import re
import struct
import urllib.parse
import urllib.request
import zlib
from .datetimes import _parse_date
from .urls import convert_to_idn
class _FeedURLHandler(urllib.request.HTTPDigestAuthHandler, urllib.request.HTTPRedirectHandler, urllib.request.HTTPDefaultErrorHandler):

    def http_error_default(self, req, fp, code, msg, headers):
        fp.status = code
        return fp

    def http_error_301(self, req, fp, code, msg, hdrs):
        result = urllib.request.HTTPRedirectHandler.http_error_301(self, req, fp, code, msg, hdrs)
        if not result:
            return fp
        result.status = code
        result.newurl = result.geturl()
        return result
    http_error_300 = http_error_301
    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301

    def http_error_401(self, req, fp, code, msg, headers):
        host = urllib.parse.urlparse(req.get_full_url())[1]
        if 'Authorization' not in req.headers or 'WWW-Authenticate' not in headers:
            return self.http_error_default(req, fp, code, msg, headers)
        auth = base64.decodebytes(req.headers['Authorization'].split(' ')[1].encode()).decode()
        user, passw = auth.split(':')
        realm = re.findall('realm="([^"]*)"', headers['WWW-Authenticate'])[0]
        self.add_password(realm, host, user, passw)
        retry = self.http_error_auth_reqed('www-authenticate', host, req, headers)
        self.reset_retry_count()
        return retry