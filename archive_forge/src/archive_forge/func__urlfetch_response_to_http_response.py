from __future__ import absolute_import
import io
import logging
import warnings
from ..exceptions import (
from ..packages.six.moves.urllib.parse import urljoin
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.retry import Retry
from ..util.timeout import Timeout
from . import _appengine_environ
def _urlfetch_response_to_http_response(self, urlfetch_resp, **response_kw):
    if is_prod_appengine():
        content_encoding = urlfetch_resp.headers.get('content-encoding')
        if content_encoding == 'deflate':
            del urlfetch_resp.headers['content-encoding']
    transfer_encoding = urlfetch_resp.headers.get('transfer-encoding')
    if transfer_encoding == 'chunked':
        encodings = transfer_encoding.split(',')
        encodings.remove('chunked')
        urlfetch_resp.headers['transfer-encoding'] = ','.join(encodings)
    original_response = HTTPResponse(body=io.BytesIO(urlfetch_resp.content), msg=urlfetch_resp.header_msg, headers=urlfetch_resp.headers, status=urlfetch_resp.status_code, **response_kw)
    return HTTPResponse(body=io.BytesIO(urlfetch_resp.content), headers=urlfetch_resp.headers, status=urlfetch_resp.status_code, original_response=original_response, **response_kw)