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
class DummyWarningsModule:
    """
    Stub for L{warnings} which provides a C{showwarning} method that is a no-op.
    """

    def showwarning(*args: Any, **kwargs: Any) -> None:
        """
        Do nothing.

        @param args: ignored.
        @param kwargs: ignored.
        """