import SOAPpy
from twisted.internet import defer
from twisted.web import client, resource, server
class SOAPPublisher(resource.Resource):
    """Publish SOAP methods.

    By default, publish methods beginning with 'soap_'. If the method
    has an attribute 'useKeywords', it well get the arguments passed
    as keyword args.
    """
    isLeaf = 1
    encoding = 'UTF-8'

    def lookupFunction(self, functionName):
        """Lookup published SOAP function.

        Override in subclasses. Default behaviour - publish methods
        starting with soap_.

        @return: callable or None if not found.
        """
        return getattr(self, 'soap_%s' % functionName, None)

    def render(self, request):
        """Handle a SOAP command."""
        data = request.content.read()
        p, header, body, attrs = SOAPpy.parseSOAPRPC(data, 1, 1, 1)
        methodName, args, kwargs = (p._name, p._aslist, p._asdict)
        if callable(args):
            args = args()
        if callable(kwargs):
            kwargs = kwargs()
        function = self.lookupFunction(methodName)
        if not function:
            self._methodNotFound(request, methodName)
            return server.NOT_DONE_YET
        elif hasattr(function, 'useKeywords'):
            keywords = {}
            for k, v in kwargs.items():
                keywords[str(k)] = v
            d = defer.maybeDeferred(function, **keywords)
        else:
            d = defer.maybeDeferred(function, *args)
        d.addCallback(self._gotResult, request, methodName)
        d.addErrback(self._gotError, request, methodName)
        return server.NOT_DONE_YET

    def _methodNotFound(self, request, methodName):
        response = SOAPpy.buildSOAP(SOAPpy.faultType('%s:Client' % SOAPpy.NS.ENV_T, 'Method %s not found' % methodName), encoding=self.encoding)
        self._sendResponse(request, response, status=500)

    def _gotResult(self, result, request, methodName):
        if not isinstance(result, SOAPpy.voidType):
            result = {'Result': result}
        response = SOAPpy.buildSOAP(kw={'%sResponse' % methodName: result}, encoding=self.encoding)
        self._sendResponse(request, response)

    def _gotError(self, failure, request, methodName):
        e = failure.value
        if isinstance(e, SOAPpy.faultType):
            fault = e
        else:
            fault = SOAPpy.faultType('%s:Server' % SOAPpy.NS.ENV_T, 'Method %s failed.' % methodName)
        response = SOAPpy.buildSOAP(fault, encoding=self.encoding)
        self._sendResponse(request, response, status=500)

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