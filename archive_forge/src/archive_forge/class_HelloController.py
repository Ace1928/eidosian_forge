import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
class HelloController(helper.Controller):
    """Controller for serving WSGI apps."""

    def hello(req, resp):
        """Render Hello world."""
        return 'Hello world!'

    def body_required(req, resp):
        """Render Hello world or set 411."""
        if req.environ.get('Content-Length', None) is None:
            resp.status = '411 Length Required'
            return
        return 'Hello world!'

    def query_string(req, resp):
        """Render QUERY_STRING value."""
        return req.environ.get('QUERY_STRING', '')

    def asterisk(req, resp):
        """Render request method value."""
        method = req.environ.get('REQUEST_METHOD', 'NO METHOD FOUND')
        tmpl = 'Got asterisk URI path with {method} method'
        return tmpl.format(**locals())

    def _munge(string):
        """Encode PATH_INFO correctly depending on Python version.

        WSGI 1.0 is a mess around unicode. Create endpoints
        that match the PATH_INFO that it produces.
        """
        return string.encode('utf-8').decode('latin-1')
    handlers = {'/hello': hello, '/no_body': hello, '/body_required': body_required, '/query_string': query_string, _munge('/привіт'): hello, _munge('/Юххууу'): hello, '/\xa0Ðblah key 0 900 4 data': hello, '/*': asterisk}