import hmac
import time
import base64
import hashlib
from io import FileIO as file
from libcloud.utils.py3 import b, next, httplib, urlparse, urlquote, urlencode, urlunquote
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError
from libcloud.storage.base import CHUNK_SIZE, Object, Container, StorageDriver
from libcloud.storage.types import (
class AtmosConnection(ConnectionUserAndKey):
    responseCls = AtmosResponse

    def add_default_headers(self, headers):
        headers['x-emc-uid'] = self.user_id
        headers['Date'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
        headers['x-emc-date'] = headers['Date']
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/octet-stream'
        if 'Accept' not in headers:
            headers['Accept'] = '*/*'
        return headers

    def pre_connect_hook(self, params, headers):
        headers['x-emc-signature'] = self._calculate_signature(params, headers)
        return (params, headers)

    def _calculate_signature(self, params, headers):
        pathstring = urlunquote(self.action)
        driver_path = self.driver.path
        if pathstring.startswith(driver_path):
            pathstring = pathstring[len(driver_path):]
        if params:
            if type(params) is dict:
                params = list(params.items())
            pathstring += '?' + urlencode(params)
        pathstring = pathstring.lower()
        xhdrs = [(k, v) for k, v in list(headers.items()) if k.startswith('x-emc-')]
        xhdrs.sort(key=lambda x: x[0])
        signature = [self.method, headers.get('Content-Type', ''), headers.get('Range', ''), headers.get('Date', ''), pathstring]
        signature.extend([k + ':' + collapse(v) for k, v in xhdrs])
        signature = '\n'.join(signature)
        key = base64.b64decode(self.key)
        signature = hmac.new(b(key), b(signature), hashlib.sha1).digest()
        return base64.b64encode(b(signature)).decode('utf-8')