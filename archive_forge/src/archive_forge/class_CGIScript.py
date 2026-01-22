import os
import urllib
from typing import AnyStr
from twisted.internet import protocol
from twisted.logger import Logger
from twisted.python import filepath
from twisted.spread import pb
from twisted.web import http, resource, server, static
class CGIScript(resource.Resource):
    """
    L{CGIScript} is a resource which runs child processes according to the CGI
    specification.

    The implementation is complex due to the fact that it requires asynchronous
    IPC with an external process with an unpleasant protocol.
    """
    isLeaf = 1

    def __init__(self, filename, registry=None, reactor=None):
        """
        Initialize, with the name of a CGI script file.
        """
        self.filename = filename
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor

    def render(self, request):
        """
        Do various things to conform to the CGI specification.

        I will set up the usual slew of environment variables, then spin off a
        process.

        @type request: L{twisted.web.http.Request}
        @param request: An HTTP request.
        """
        scriptName = b'/' + b'/'.join(request.prepath)
        serverName = request.getRequestHostname().split(b':')[0]
        env = {'SERVER_SOFTWARE': server.version, 'SERVER_NAME': serverName, 'GATEWAY_INTERFACE': 'CGI/1.1', 'SERVER_PROTOCOL': request.clientproto, 'SERVER_PORT': str(request.getHost().port), 'REQUEST_METHOD': request.method, 'SCRIPT_NAME': scriptName, 'SCRIPT_FILENAME': self.filename, 'REQUEST_URI': request.uri}
        ip = request.getClientAddress().host
        if ip is not None:
            env['REMOTE_ADDR'] = ip
        pp = request.postpath
        if pp:
            env['PATH_INFO'] = '/' + '/'.join(pp)
        if hasattr(request, 'content'):
            request.content.seek(0, 2)
            length = request.content.tell()
            request.content.seek(0, 0)
            env['CONTENT_LENGTH'] = str(length)
        try:
            qindex = request.uri.index(b'?')
        except ValueError:
            env['QUERY_STRING'] = ''
            qargs = []
        else:
            qs = env['QUERY_STRING'] = request.uri[qindex + 1:]
            if b'=' in qs:
                qargs = []
            else:
                qargs = [urllib.parse.unquote(x.decode()) for x in qs.split(b'+')]
        for title, header in request.getAllHeaders().items():
            envname = title.replace(b'-', b'_').upper()
            if title not in (b'content-type', b'content-length', b'proxy'):
                envname = b'HTTP_' + envname
            env[envname] = header
        for key, value in os.environ.items():
            if key not in env:
                env[key] = value
        self.runProcess(env, request, qargs)
        return server.NOT_DONE_YET

    def runProcess(self, env, request, qargs=[]):
        """
        Run the cgi script.

        @type env: A L{dict} of L{str}, or L{None}
        @param env: The environment variables to pass to the process that will
            get spawned. See
            L{twisted.internet.interfaces.IReactorProcess.spawnProcess} for
            more information about environments and process creation.

        @type request: L{twisted.web.http.Request}
        @param request: An HTTP request.

        @type qargs: A L{list} of L{str}
        @param qargs: The command line arguments to pass to the process that
            will get spawned.
        """
        p = CGIProcessProtocol(request)
        self._reactor.spawnProcess(p, self.filename, [self.filename] + qargs, env, os.path.dirname(self.filename))