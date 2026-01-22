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
class MultiCapture(Generic[AnyStr]):
    _state = None
    _in_suspended = False

    def __init__(self, in_: Optional[CaptureBase[AnyStr]], out: Optional[CaptureBase[AnyStr]], err: Optional[CaptureBase[AnyStr]]) -> None:
        self.in_: Optional[CaptureBase[AnyStr]] = in_
        self.out: Optional[CaptureBase[AnyStr]] = out
        self.err: Optional[CaptureBase[AnyStr]] = err

    def __repr__(self) -> str:
        return '<MultiCapture out={!r} err={!r} in_={!r} _state={!r} _in_suspended={!r}>'.format(self.out, self.err, self.in_, self._state, self._in_suspended)

    def start_capturing(self) -> None:
        self._state = 'started'
        if self.in_:
            self.in_.start()
        if self.out:
            self.out.start()
        if self.err:
            self.err.start()

    def pop_outerr_to_orig(self) -> Tuple[AnyStr, AnyStr]:
        """Pop current snapshot out/err capture and flush to orig streams."""
        out, err = self.readouterr()
        if out:
            assert self.out is not None
            self.out.writeorg(out)
        if err:
            assert self.err is not None
            self.err.writeorg(err)
        return (out, err)

    def suspend_capturing(self, in_: bool=False) -> None:
        self._state = 'suspended'
        if self.out:
            self.out.suspend()
        if self.err:
            self.err.suspend()
        if in_ and self.in_:
            self.in_.suspend()
            self._in_suspended = True

    def resume_capturing(self) -> None:
        self._state = 'started'
        if self.out:
            self.out.resume()
        if self.err:
            self.err.resume()
        if self._in_suspended:
            assert self.in_ is not None
            self.in_.resume()
            self._in_suspended = False

    def stop_capturing(self) -> None:
        """Stop capturing and reset capturing streams."""
        if self._state == 'stopped':
            raise ValueError('was already stopped')
        self._state = 'stopped'
        if self.out:
            self.out.done()
        if self.err:
            self.err.done()
        if self.in_:
            self.in_.done()

    def is_started(self) -> bool:
        """Whether actively capturing -- not suspended or stopped."""
        return self._state == 'started'

    def readouterr(self) -> CaptureResult[AnyStr]:
        out = self.out.snap() if self.out else ''
        err = self.err.snap() if self.err else ''
        return CaptureResult(out, err)