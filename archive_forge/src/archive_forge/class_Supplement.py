import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
class Supplement(object):
    """
    This is a supplement used to display standard WSGI information in
    the traceback.
    """

    def __init__(self, middleware, environ):
        self.middleware = middleware
        self.environ = environ
        self.source_url = request.construct_url(environ)

    def extraData(self):
        data = {}
        cgi_vars = data['extra', 'CGI Variables'] = {}
        wsgi_vars = data['extra', 'WSGI Variables'] = {}
        hide_vars = ['paste.config', 'wsgi.errors', 'wsgi.input', 'wsgi.multithread', 'wsgi.multiprocess', 'wsgi.run_once', 'wsgi.version', 'wsgi.url_scheme']
        for name, value in self.environ.items():
            if name.upper() == name:
                if value:
                    cgi_vars[name] = value
            elif name not in hide_vars:
                wsgi_vars[name] = value
        if self.environ['wsgi.version'] != (1, 0):
            wsgi_vars['wsgi.version'] = self.environ['wsgi.version']
        proc_desc = tuple([int(bool(self.environ[key])) for key in ('wsgi.multiprocess', 'wsgi.multithread', 'wsgi.run_once')])
        wsgi_vars['wsgi process'] = self.process_combos[proc_desc]
        wsgi_vars['application'] = self.middleware.application
        if 'paste.config' in self.environ:
            data['extra', 'Configuration'] = dict(self.environ['paste.config'])
        return data
    process_combos = {(0, 0, 0): 'Non-concurrent server', (0, 1, 0): 'Multithreaded', (1, 0, 0): 'Multiprocess', (1, 1, 0): 'Multi process AND threads (?)', (0, 0, 1): 'Non-concurrent CGI', (0, 1, 1): 'Multithread CGI (?)', (1, 0, 1): 'CGI', (1, 1, 1): 'Multi thread/process CGI (?)'}