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
class DummyPIDFile(NonePIDFile):
    """
    Stub for L{PIDFile}.

    Tracks context manager entry/exit without doing anything.
    """

    def __init__(self) -> None:
        NonePIDFile.__init__(self)
        self.entered = False
        self.exited = False

    def __enter__(self) -> 'DummyPIDFile':
        self.entered = True
        return self

    def __exit__(self, excType: Optional[Type[BaseException]], excValue: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.exited = True