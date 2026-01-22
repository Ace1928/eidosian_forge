import SOAPpy
from twisted.internet import defer
from twisted.web import client, resource, server
def _gotResult(self, result, request, methodName):
    if not isinstance(result, SOAPpy.voidType):
        result = {'Result': result}
    response = SOAPpy.buildSOAP(kw={'%sResponse' % methodName: result}, encoding=self.encoding)
    self._sendResponse(request, response)