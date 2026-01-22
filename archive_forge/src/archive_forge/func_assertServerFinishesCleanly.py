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
def assertServerFinishesCleanly(self, process):
    """Shutdown the brz serve instance process looking for errors."""
    result = self.finish_brz_subprocess(process, retcode=3, send_signal=signal.SIGINT)
    self.assertEqual(b'', result[0])
    self.assertEqual(b'brz: interrupted\n', result[1])