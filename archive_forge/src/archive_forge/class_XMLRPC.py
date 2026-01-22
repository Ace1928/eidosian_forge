import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
class XMLRPC(resource.Resource):
    """
    A resource that implements XML-RPC.

    You probably want to connect this to '/RPC2'.

    Methods published can return XML-RPC serializable results, Faults,
    Binary, Boolean, DateTime, Deferreds, or Handler instances.

    By default methods beginning with 'xmlrpc_' are published.

    Sub-handlers for prefixed methods (e.g., system.listMethods)
    can be added with putSubHandler. By default, prefixes are
    separated with a '.'. Override self.separator to change this.

    @ivar allowNone: Permit XML translating of Python constant None.
    @type allowNone: C{bool}

    @ivar useDateTime: Present C{datetime} values as C{datetime.datetime}
        objects?
    @type useDateTime: C{bool}
    """
    NOT_FOUND = 8001
    FAILURE = 8002
    isLeaf = 1
    separator = '.'
    allowedMethods = (b'POST',)
    _log = Logger()

    def __init__(self, allowNone=False, useDateTime=False):
        resource.Resource.__init__(self)
        self.subHandlers = {}
        self.allowNone = allowNone
        self.useDateTime = useDateTime

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def putSubHandler(self, prefix, handler):
        self.subHandlers[prefix] = handler

    def getSubHandler(self, prefix):
        return self.subHandlers.get(prefix, None)

    def getSubHandlerPrefixes(self):
        return list(self.subHandlers.keys())

    def render_POST(self, request):
        request.content.seek(0, 0)
        request.setHeader(b'content-type', b'text/xml; charset=utf-8')
        try:
            args, functionPath = xmlrpclib.loads(request.content.read(), use_datetime=self.useDateTime)
        except Exception as e:
            f = Fault(self.FAILURE, f"Can't deserialize input: {e}")
            self._cbRender(f, request)
        else:
            try:
                function = self.lookupProcedure(functionPath)
            except Fault as f:
                self._cbRender(f, request)
            else:
                responseFailed = []
                request.notifyFinish().addErrback(responseFailed.append)
                if getattr(function, 'withRequest', False):
                    d = defer.maybeDeferred(function, request, *args)
                else:
                    d = defer.maybeDeferred(function, *args)
                d.addErrback(self._ebRender)
                d.addCallback(self._cbRender, request, responseFailed)
        return server.NOT_DONE_YET

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

    def _ebRender(self, failure):
        if isinstance(failure.value, Fault):
            return failure.value
        self._log.failure('', failure)
        return Fault(self.FAILURE, 'error')

    def lookupProcedure(self, procedurePath):
        """
        Given a string naming a procedure, return a callable object for that
        procedure or raise NoSuchFunction.

        The returned object will be called, and should return the result of the
        procedure, a Deferred, or a Fault instance.

        Override in subclasses if you want your own policy.  The base
        implementation that given C{'foo'}, C{self.xmlrpc_foo} will be returned.
        If C{procedurePath} contains C{self.separator}, the sub-handler for the
        initial prefix is used to search for the remaining path.

        If you override C{lookupProcedure}, you may also want to override
        C{listProcedures} to accurately report the procedures supported by your
        resource, so that clients using the I{system.listMethods} procedure
        receive accurate results.

        @since: 11.1
        """
        if procedurePath.find(self.separator) != -1:
            prefix, procedurePath = procedurePath.split(self.separator, 1)
            handler = self.getSubHandler(prefix)
            if handler is None:
                raise NoSuchFunction(self.NOT_FOUND, 'no such subHandler %s' % prefix)
            return handler.lookupProcedure(procedurePath)
        f = getattr(self, 'xmlrpc_%s' % procedurePath, None)
        if not f:
            raise NoSuchFunction(self.NOT_FOUND, 'procedure %s not found' % procedurePath)
        elif not callable(f):
            raise NoSuchFunction(self.NOT_FOUND, 'procedure %s not callable' % procedurePath)
        else:
            return f

    def listProcedures(self):
        """
        Return a list of the names of all xmlrpc procedures.

        @since: 11.1
        """
        return reflect.prefixedMethodNames(self.__class__, 'xmlrpc_')