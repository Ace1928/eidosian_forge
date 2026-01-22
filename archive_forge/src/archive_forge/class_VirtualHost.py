import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
class VirtualHost(object):
    """Select a different WSGI application based on the Host header.

    This can be useful when running multiple sites within one CP server.
    It allows several domains to point to different applications. For example::

        root = Root()
        RootApp = cherrypy.Application(root)
        Domain2App = cherrypy.Application(root)
        SecureApp = cherrypy.Application(Secure())

        vhost = cherrypy._cpwsgi.VirtualHost(
            RootApp,
            domains={
                'www.domain2.example': Domain2App,
                'www.domain2.example:443': SecureApp,
            },
        )

        cherrypy.tree.graft(vhost)
    """
    default = None
    'Required. The default WSGI application.'
    use_x_forwarded_host = True
    'If True (the default), any "X-Forwarded-Host"\n    request header will be used instead of the "Host" header. This\n    is commonly added by HTTP servers (such as Apache) when proxying.'
    domains = {}
    'A dict of {host header value: application} pairs.\n    The incoming "Host" request header is looked up in this dict,\n    and, if a match is found, the corresponding WSGI application\n    will be called instead of the default. Note that you often need\n    separate entries for "example.com" and "www.example.com".\n    In addition, "Host" headers may contain the port number.\n    '

    def __init__(self, default, domains=None, use_x_forwarded_host=True):
        self.default = default
        self.domains = domains or {}
        self.use_x_forwarded_host = use_x_forwarded_host

    def __call__(self, environ, start_response):
        domain = environ.get('HTTP_HOST', '')
        if self.use_x_forwarded_host:
            domain = environ.get('HTTP_X_FORWARDED_HOST', domain)
        nextapp = self.domains.get(domain)
        if nextapp is None:
            nextapp = self.default
        return nextapp(environ, start_response)