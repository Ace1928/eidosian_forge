import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
class TestInetServer(tests.TestCase):

    def create_file_pipes(self):
        r, w = os.pipe()
        rf = os.fdopen(r, 'rb')
        wf = os.fdopen(w, 'wb')
        return (rf, wf)

    def test_inet_server_responds_to_sighup(self):
        t = transport.get_transport('memory:///')
        content = b'a' * 1024 * 1024
        t.put_bytes('bigfile', content)
        factory = server.BzrServerFactory()
        client_read, server_write = self.create_file_pipes()
        server_read, client_write = self.create_file_pipes()
        factory._get_stdin_stdout = lambda: (server_read, server_write)
        factory.set_up(t, None, None, inet=True, timeout=4.0)
        self.addCleanup(factory.tear_down)
        started = threading.Event()
        stopped = threading.Event()

        def serving():
            started.set()
            factory.smart_server.serve()
            stopped.set()
        server_thread = threading.Thread(target=serving)
        server_thread.start()
        started.wait()
        client_medium = medium.SmartSimplePipesClientMedium(client_read, client_write, 'base')
        client_client = client._SmartClient(client_medium)
        resp, response_handler = client_client.call_expecting_body(b'get', b'bigfile')
        signals._sighup_handler(SIGHUP, None)
        self.assertTrue(factory.smart_server.finished)
        v = response_handler.read_body_bytes()
        if v != content:
            self.fail('Got the wrong content back, expected 1M "a"')
        stopped.wait()
        server_thread.join()