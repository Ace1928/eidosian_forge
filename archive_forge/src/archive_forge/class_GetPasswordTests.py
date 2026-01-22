import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
@skipIf(not IReactorProcess.providedBy(reactor), 'Process support required to test getPassword')
class GetPasswordTests(TestCase):

    def test_stdin(self):
        """
        Making sure getPassword accepts a password from standard input by
        running a child process which uses getPassword to read in a string
        which it then writes it out again.  Write a string to the child
        process and then read one and make sure it is the right string.
        """
        p = PasswordTestingProcessProtocol()
        p.finished = Deferred()
        reactor.spawnProcess(p, pyExe, [pyExe, b'-c', b'import sys\nfrom twisted.python.util import getPassword\nsys.stdout.write(getPassword())\nsys.stdout.flush()\n'], env={b'PYTHONPATH': os.pathsep.join(sys.path).encode('utf8')})

        def processFinished(result):
            reason, output = result
            reason.trap(ProcessDone)
            self.assertIn((1, b'secret'), output)
        return p.finished.addCallback(processFinished)