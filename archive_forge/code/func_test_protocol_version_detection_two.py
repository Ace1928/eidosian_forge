from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_protocol_version_detection_two(self):
    transport = memory.MemoryTransport()
    wsgi_app = wsgi.SmartWSGIApp(transport)
    fake_input = BytesIO(protocol.REQUEST_VERSION_TWO + b'hello\n')
    environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo'})
    iterable = wsgi_app(environ, self.start_response)
    response = self.read_response(iterable)
    self.assertEqual('200 OK', self.status)
    self.assertEqual(protocol.RESPONSE_VERSION_TWO + b'success\nok\x012\n', response)