import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
class PTYProcessTestsBuilder(ProcessTestsBuilderBase):
    """
    Builder defining tests relating to L{IReactorProcess} for child processes
    which have a PTY.
    """
    usePTY = True
    if platform.isWindows():
        skip = 'PTYs are not supported on Windows.'
    elif platform.isMacOSX():
        skip = 'PTYs are flaky from a Darwin bug. See #8840.'
        skippedReactors = {'twisted.internet.pollreactor.PollReactor': "macOS's poll() does not support PTYs"}