from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
class WSGITestMixin:

    def build_environ(self, updates=None):
        """Builds an environ dict with all fields required by PEP 333.

        :param updates: a dict to that will be incorporated into the returned
            dict using dict.update(updates).
        """
        environ = {'REQUEST_METHOD': 'GET', 'SCRIPT_NAME': '/script/name/', 'PATH_INFO': 'path/info', 'SERVER_NAME': 'test', 'SERVER_PORT': '9999', 'SERVER_PROTOCOL': 'HTTP/1.0', 'wsgi.version': (1, 0), 'wsgi.url_scheme': 'http', 'wsgi.input': BytesIO(b''), 'wsgi.errors': BytesIO(), 'wsgi.multithread': False, 'wsgi.multiprocess': False, 'wsgi.run_once': True}
        if updates is not None:
            environ.update(updates)
        return environ

    def read_response(self, iterable):
        response = b''
        for string in iterable:
            response += string
        return response

    def start_response(self, status, headers):
        self.status = status
        self.headers = headers