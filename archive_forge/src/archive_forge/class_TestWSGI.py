from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
class TestWSGI(tests.TestCaseInTempDir, WSGITestMixin):

    def setUp(self):
        super().setUp()
        self.status = None
        self.headers = None

    def test_construct(self):
        app = wsgi.SmartWSGIApp(FakeTransport())
        self.assertIsInstance(app.backing_transport, chroot.ChrootTransport)

    def test_http_get_rejected(self):
        app = wsgi.SmartWSGIApp(FakeTransport())
        environ = self.build_environ({'REQUEST_METHOD': 'GET'})
        iterable = app(environ, self.start_response)
        self.read_response(iterable)
        self.assertEqual('405 Method not allowed', self.status)
        self.assertTrue(('Allow', 'POST') in self.headers)

    def _fake_make_request(self, transport, write_func, bytes, rcp):
        request = FakeRequest(transport, write_func)
        request.accept_bytes(bytes)
        self.request = request
        return request

    def test_smart_wsgi_app_uses_given_relpath(self):
        transport = FakeTransport()
        wsgi_app = wsgi.SmartWSGIApp(transport)
        wsgi_app.backing_transport = transport
        wsgi_app.make_request = self._fake_make_request
        fake_input = BytesIO(b'fake request')
        environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo/bar'})
        iterable = wsgi_app(environ, self.start_response)
        response = self.read_response(iterable)
        self.assertEqual([('clone', 'foo/bar/')], transport.calls)

    def test_smart_wsgi_app_request_and_response(self):
        transport = memory.MemoryTransport()
        transport.put_bytes('foo', b'some bytes')
        wsgi_app = wsgi.SmartWSGIApp(transport)
        wsgi_app.make_request = self._fake_make_request
        fake_input = BytesIO(b'fake request')
        environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo'})
        iterable = wsgi_app(environ, self.start_response)
        response = self.read_response(iterable)
        self.assertEqual('200 OK', self.status)
        self.assertEqual(b'got bytes: fake request', response)

    def test_relpath_setter(self):
        calls = []

        def fake_app(environ, start_response):
            calls.append(environ['breezy.relpath'])
        wrapped_app = wsgi.RelpathSetter(fake_app, prefix='/abc/', path_var='FOO')
        wrapped_app({'FOO': '/abc/xyz/.bzr/smart'}, None)
        self.assertEqual(['xyz'], calls)

    def test_relpath_setter_bad_path_prefix(self):

        def fake_app(environ, start_response):
            self.fail('The app should never be called when the path is wrong')
        wrapped_app = wsgi.RelpathSetter(fake_app, prefix='/abc/', path_var='FOO')
        iterable = wrapped_app({'FOO': 'AAA/abc/xyz/.bzr/smart'}, self.start_response)
        self.read_response(iterable)
        self.assertTrue(self.status.startswith('404'))

    def test_relpath_setter_bad_path_suffix(self):

        def fake_app(environ, start_response):
            self.fail('The app should never be called when the path is wrong')
        wrapped_app = wsgi.RelpathSetter(fake_app, prefix='/abc/', path_var='FOO')
        iterable = wrapped_app({'FOO': '/abc/xyz/.bzr/AAA'}, self.start_response)
        self.read_response(iterable)
        self.assertTrue(self.status.startswith('404'))

    def test_make_app(self):
        app = wsgi.make_app(root='a root', prefix='a prefix', path_var='a path_var')
        self.assertIsInstance(app, wsgi.RelpathSetter)
        self.assertIsInstance(app.app, wsgi.SmartWSGIApp)
        self.assertStartsWith(app.app.backing_transport.base, 'chroot-')
        backing_transport = app.app.backing_transport
        chroot_backing_transport = backing_transport.server.backing_transport
        self.assertEndsWith(chroot_backing_transport.base, 'a%20root/')
        self.assertEqual(app.app.root_client_path, 'a prefix')
        self.assertEqual(app.path_var, 'a path_var')

    def test_incomplete_request(self):
        transport = FakeTransport()
        wsgi_app = wsgi.SmartWSGIApp(transport)

        def make_request(transport, write_func, bytes, root_client_path):
            request = IncompleteRequest(transport, write_func)
            request.accept_bytes(bytes)
            self.request = request
            return request
        wsgi_app.make_request = make_request
        fake_input = BytesIO(b'incomplete request')
        environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo/bar'})
        iterable = wsgi_app(environ, self.start_response)
        response = self.read_response(iterable)
        self.assertEqual('200 OK', self.status)
        self.assertEqual(b'error\x01incomplete request\n', response)

    def test_protocol_version_detection_one(self):
        transport = memory.MemoryTransport()
        wsgi_app = wsgi.SmartWSGIApp(transport)
        fake_input = BytesIO(b'hello\n')
        environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo'})
        iterable = wsgi_app(environ, self.start_response)
        response = self.read_response(iterable)
        self.assertEqual('200 OK', self.status)
        self.assertEqual(b'ok\x012\n', response)

    def test_protocol_version_detection_two(self):
        transport = memory.MemoryTransport()
        wsgi_app = wsgi.SmartWSGIApp(transport)
        fake_input = BytesIO(protocol.REQUEST_VERSION_TWO + b'hello\n')
        environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo'})
        iterable = wsgi_app(environ, self.start_response)
        response = self.read_response(iterable)
        self.assertEqual('200 OK', self.status)
        self.assertEqual(protocol.RESPONSE_VERSION_TWO + b'success\nok\x012\n', response)