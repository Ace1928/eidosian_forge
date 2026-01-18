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
def test_startReactorWhenRunning(self) -> None:
    """
        L{Runner.startReactor} ensures that C{whenRunning} is called with
        C{whenRunningArguments} when the reactor is running.
        """
    self._testHook('whenRunning', 'startReactor')