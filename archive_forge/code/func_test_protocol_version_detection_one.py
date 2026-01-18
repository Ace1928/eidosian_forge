from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_protocol_version_detection_one(self):
    transport = memory.MemoryTransport()
    wsgi_app = wsgi.SmartWSGIApp(transport)
    fake_input = BytesIO(b'hello\n')
    environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo'})
    iterable = wsgi_app(environ, self.start_response)
    response = self.read_response(iterable)
    self.assertEqual('200 OK', self.status)
    self.assertEqual(b'ok\x012\n', response)