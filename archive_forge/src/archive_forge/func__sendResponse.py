import SOAPpy
from twisted.internet import defer
from twisted.web import client, resource, server
def _sendResponse(self, request, response, status=200):
    request.setResponseCode(status)
    if self.encoding is not None:
        mimeType = 'text/xml; charset="%s"' % self.encoding
    else:
        mimeType = 'text/xml'
    request.setHeader('Content-type', mimeType)
    request.setHeader('Content-length', str(len(response)))
    request.write(response)
    request.finish()