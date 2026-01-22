import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def _cbRender(self, result, request, responseFailed=None):
    if responseFailed:
        return
    if isinstance(result, Handler):
        result = result.result
    if not isinstance(result, Fault):
        result = (result,)
    try:
        try:
            content = xmlrpclib.dumps(result, methodresponse=True, allow_none=self.allowNone)
        except Exception as e:
            f = Fault(self.FAILURE, f"Can't serialize output: {e}")
            content = xmlrpclib.dumps(f, methodresponse=True, allow_none=self.allowNone)
        if isinstance(content, str):
            content = content.encode('utf8')
        request.setHeader(b'content-length', b'%d' % (len(content),))
        request.write(content)
    except Exception:
        self._log.failure('')
    request.finish()