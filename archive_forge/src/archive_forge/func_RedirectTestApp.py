import http.client as http
import eventlet.patcher
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def RedirectTestApp(name):

    class App(object):
        """
        Test WSGI application which can respond with multiple kinds of HTTP
        redirects and is used to verify Glance client redirects.
        """

        def __init__(self):
            """
            Initialize app with a name and port.
            """
            self.name = name

        @webob.dec.wsgify
        def __call__(self, request):
            """
            Handles all requests to the application.
            """
            base = 'http://%s' % request.host
            path = request.path_qs
            if path == '/':
                return 'root'
            elif path == '/302':
                url = '%s/success' % base
                raise webob.exc.HTTPFound(location=url)
            elif path == '/302?with_qs=yes':
                url = '%s/success?with_qs=yes' % base
                raise webob.exc.HTTPFound(location=url)
            elif path == '/infinite_302':
                raise webob.exc.HTTPFound(location=request.url)
            elif path.startswith('/redirect-to'):
                url = 'http://127.0.0.1:%s/success' % path.split('-')[-1]
                raise webob.exc.HTTPFound(location=url)
            elif path == '/success':
                return 'success_from_host_%s' % self.name
            elif path == '/success?with_qs=yes':
                return 'success_with_qs'
            return 'fail'
    return App