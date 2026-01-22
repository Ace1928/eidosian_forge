import sys
import threading
from typing import Tuple
from wsgiref import simple_server
from dulwich.tests import SkipTest, skipIf
from ...server import DictBackend, ReceivePackHandler, UploadPackHandler
from ...web import (
from .server_utils import NoSideBand64kReceivePackHandler, ServerTests
from .utils import CompatTestCase
def _check_app(self, app):
    receive_pack_handler_cls = app.handlers[b'git-receive-pack']
    caps = receive_pack_handler_cls.capabilities()
    self.assertIn(b'side-band-64k', caps)
    self.assertIn(b'no-done', caps)