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
class SysCaptureBase(CaptureBase[AnyStr]):

    def __init__(self, fd: int, tmpfile: Optional[TextIO]=None, *, tee: bool=False) -> None:
        name = patchsysdict[fd]
        self._old: TextIO = getattr(sys, name)
        self.name = name
        if tmpfile is None:
            if name == 'stdin':
                tmpfile = DontReadFromInput()
            else:
                tmpfile = CaptureIO() if not tee else TeeCaptureIO(self._old)
        self.tmpfile = tmpfile
        self._state = 'initialized'

    def repr(self, class_name: str) -> str:
        return '<{} {} _old={} _state={!r} tmpfile={!r}>'.format(class_name, self.name, hasattr(self, '_old') and repr(self._old) or '<UNSET>', self._state, self.tmpfile)

    def __repr__(self) -> str:
        return '<{} {} _old={} _state={!r} tmpfile={!r}>'.format(self.__class__.__name__, self.name, hasattr(self, '_old') and repr(self._old) or '<UNSET>', self._state, self.tmpfile)

    def _assert_state(self, op: str, states: Tuple[str, ...]) -> None:
        assert self._state in states, 'cannot {} in state {!r}: expected one of {}'.format(op, self._state, ', '.join(states))

    def start(self) -> None:
        self._assert_state('start', ('initialized',))
        setattr(sys, self.name, self.tmpfile)
        self._state = 'started'

    def done(self) -> None:
        self._assert_state('done', ('initialized', 'started', 'suspended', 'done'))
        if self._state == 'done':
            return
        setattr(sys, self.name, self._old)
        del self._old
        self.tmpfile.close()
        self._state = 'done'

    def suspend(self) -> None:
        self._assert_state('suspend', ('started', 'suspended'))
        setattr(sys, self.name, self._old)
        self._state = 'suspended'

    def resume(self) -> None:
        self._assert_state('resume', ('started', 'suspended'))
        if self._state == 'started':
            return
        setattr(sys, self.name, self.tmpfile)
        self._state = 'started'