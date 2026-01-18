import errno
from io import StringIO
from signal import SIGTERM
from types import TracebackType
from typing import Any, Iterable, List, Optional, TextIO, Tuple, Type, Union, cast
from attr import Factory, attrib, attrs
import twisted.trial.unittest
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.filepath import FilePath
from ...runner import _runner
from .._exit import ExitStatus
from .._pidfile import NonePIDFile, PIDFile
from .._runner import Runner
def test_runAlreadyRunning(self) -> None:
    """
        L{Runner.run} exits with L{ExitStatus.EX_USAGE} and the expected
        message if a process is already running that corresponds to the given
        PID file.
        """
    pidFile = PIDFile(self.filePath(self.pidFileContent))
    pidFile.isRunning = lambda: True
    runner = Runner(reactor=MemoryReactor(), pidFile=pidFile)
    runner.run()
    self.assertEqual(self.exit.status, ExitStatus.EX_CONFIG)
    self.assertEqual(self.exit.message, 'Already running.')