from io import BytesIO
from requests import Session
from ..client import (
from ..errors import GitProtocolError, NotGitRepository
def _http_request(self, url, headers=None, data=None, allow_compression=False):
    req_headers = self.session.headers.copy()
    if headers is not None:
        req_headers.update(headers)
    if allow_compression:
        req_headers['Accept-Encoding'] = 'gzip'
    else:
        req_headers['Accept-Encoding'] = 'identity'
    if data:
        resp = self.session.post(url, headers=req_headers, data=data)
    else:
        resp = self.session.get(url, headers=req_headers)
    if resp.status_code == 404:
        raise NotGitRepository
    if resp.status_code == 401:
        raise HTTPUnauthorized(resp.headers.get('WWW-Authenticate'), url)
    if resp.status_code == 407:
        raise HTTPProxyUnauthorized(resp.headers.get('Proxy-Authenticate'), url)
    if resp.status_code != 200:
        raise GitProtocolError('unexpected http resp %d for %s' % (resp.status_code, url))
    resp.content_type = resp.headers.get('Content-Type')
    resp.redirect_location = ''
    if resp.history:
        resp.redirect_location = resp.url
    read = BytesIO(resp.content).read
    return (resp, read)