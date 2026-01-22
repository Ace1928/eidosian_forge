import os
from shlex import quote as pquote
from xml.dom.minidom import parseString
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.py3 import ensure_string
from libcloud.utils.misc import lowercase_keys
from libcloud.common.base import LibcloudConnection, HttpLibResponseProxy
def _log_curl(self, method, url, body, headers):
    cmd = ['curl']
    if self.http_proxy_used:
        if self.proxy_username and self.proxy_password:
            proxy_url = '{}://{}:{}@{}:{}'.format(self.proxy_scheme, self.proxy_username, self.proxy_password, self.proxy_host, self.proxy_port)
        else:
            proxy_url = '{}://{}:{}'.format(self.proxy_scheme, self.proxy_host, self.proxy_port)
        proxy_url = pquote(proxy_url)
        cmd.extend(['--proxy', proxy_url])
    cmd.extend(['-i'])
    if method.lower() == 'head':
        cmd.extend(['--head'])
    else:
        cmd.extend(['-X', pquote(method)])
    for h in headers:
        cmd.extend(['-H', pquote('{}: {}'.format(h, headers[h]))])
    cert_file = getattr(self, 'cert_file', None)
    if cert_file:
        cmd.extend(['--cert', pquote(cert_file)])
    if body is not None and len(body) > 0:
        if isinstance(body, (bytearray, bytes)):
            body = body.decode('utf-8')
        cmd.extend(['--data-binary', pquote(body)])
    cmd.extend(['--compress'])
    cmd.extend([pquote('{}{}'.format(self.host, url))])
    return ' '.join(cmd)