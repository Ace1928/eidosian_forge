import sys
import threading
from typing import Tuple
from wsgiref import simple_server
from dulwich.tests import SkipTest, skipIf
from ...server import DictBackend, ReceivePackHandler, UploadPackHandler
from ...web import (
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase
@skipIf(sys.platform == 'win32', 'Broken on windows, with very long fail time.')
class WebTests(ServerTests):
    """Base tests for web server tests.

    Contains utility and setUp/tearDown methods, but does non inherit from
    TestCase so tests are not automatically run.
    """
    protocol = 'http'

    def _start_server(self, repo):
        backend = DictBackend({'/': repo})
        app = self._make_app(backend)
        dul_server = simple_server.make_server('localhost', 0, app, server_class=WSGIServerLogger, handler_class=WSGIRequestHandlerLogger)
        self.addCleanup(dul_server.shutdown)
        self.addCleanup(dul_server.server_close)
        threading.Thread(target=dul_server.serve_forever).start()
        self._server = dul_server
        _, port = dul_server.socket.getsockname()
        return port