import signal
import sys
import threading
from _thread import interrupt_main  # type: ignore
from ... import builtins, config, errors, osutils
from ... import revision as _mod_revision
from ... import trace, transport, urlutils
from ...branch import Branch
from ...bzr.smart import client, medium
from ...bzr.smart.server import BzrServerFactory, SmartTCPServer
from ...controldir import ControlDir
from ...transport import remote
from .. import TestCaseWithMemoryTransport, TestCaseWithTransport
def test_server_exception_with_hook(self):
    """Catch exception from the server in the server_exception hook.

        We use ``run_bzr_serve_then_func`` without a ``func`` so the server
        will receive a KeyboardInterrupt exception we want to catch.
        """

    def hook(exception):
        if exception[0] is KeyboardInterrupt:
            sys.stderr.write(b'catching KeyboardInterrupt\n')
            return True
        else:
            return False
    SmartTCPServer.hooks.install_named_hook('server_exception', hook, 'test_server_except_hook hook')
    args = ['--listen', 'localhost', '--port', '0', '--quiet']
    out, err = self.run_bzr_serve_then_func(args, retcode=0)
    self.assertEqual('catching KeyboardInterrupt\n', err)