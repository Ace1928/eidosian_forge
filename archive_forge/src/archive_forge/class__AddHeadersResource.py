import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
class _AddHeadersResource(resource.Resource):

    def __init__(self, originalResource, headers):
        self._originalResource = originalResource
        self._headers = headers

    def getChildWithDefault(self, name, request):
        for k, v in self._headers:
            request.responseHeaders.addRawHeader(k, v)
        return self._originalResource.getChildWithDefault(name, request)