import abc
import collections
import contextlib
import io
from io import UnsupportedOperation
import os
import sys
from tempfile import TemporaryFile
from types import TracebackType
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import SubRequest
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.reports import CollectReport
class CaptureFixture(Generic[AnyStr]):
    """Object returned by the :fixture:`capsys`, :fixture:`capsysbinary`,
    :fixture:`capfd` and :fixture:`capfdbinary` fixtures."""

    def __init__(self, captureclass: Type[CaptureBase[AnyStr]], request: SubRequest, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self.captureclass: Type[CaptureBase[AnyStr]] = captureclass
        self.request = request
        self._capture: Optional[MultiCapture[AnyStr]] = None
        self._captured_out: AnyStr = self.captureclass.EMPTY_BUFFER
        self._captured_err: AnyStr = self.captureclass.EMPTY_BUFFER

    def _start(self) -> None:
        if self._capture is None:
            self._capture = MultiCapture(in_=None, out=self.captureclass(1), err=self.captureclass(2))
            self._capture.start_capturing()

    def close(self) -> None:
        if self._capture is not None:
            out, err = self._capture.pop_outerr_to_orig()
            self._captured_out += out
            self._captured_err += err
            self._capture.stop_capturing()
            self._capture = None

    def readouterr(self) -> CaptureResult[AnyStr]:
        """Read and return the captured output so far, resetting the internal
        buffer.

        :returns:
            The captured content as a namedtuple with ``out`` and ``err``
            string attributes.
        """
        captured_out, captured_err = (self._captured_out, self._captured_err)
        if self._capture is not None:
            out, err = self._capture.readouterr()
            captured_out += out
            captured_err += err
        self._captured_out = self.captureclass.EMPTY_BUFFER
        self._captured_err = self.captureclass.EMPTY_BUFFER
        return CaptureResult(captured_out, captured_err)

    def _suspend(self) -> None:
        """Suspend this fixture's own capturing temporarily."""
        if self._capture is not None:
            self._capture.suspend_capturing()

    def _resume(self) -> None:
        """Resume this fixture's own capturing temporarily."""
        if self._capture is not None:
            self._capture.resume_capturing()

    def _is_started(self) -> bool:
        """Whether actively capturing -- not disabled or closed."""
        if self._capture is not None:
            return self._capture.is_started()
        return False

    @contextlib.contextmanager
    def disabled(self) -> Generator[None, None, None]:
        """Temporarily disable capturing while inside the ``with`` block."""
        capmanager: CaptureManager = self.request.config.pluginmanager.getplugin('capturemanager')
        with capmanager.global_and_fixture_disabled():
            yield