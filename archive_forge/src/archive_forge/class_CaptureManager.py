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
class CaptureManager:
    """The capture plugin.

    Manages that the appropriate capture method is enabled/disabled during
    collection and each test phase (setup, call, teardown). After each of
    those points, the captured output is obtained and attached to the
    collection/runtest report.

    There are two levels of capture:

    * global: enabled by default and can be suppressed by the ``-s``
      option. This is always enabled/disabled during collection and each test
      phase.

    * fixture: when a test function or one of its fixture depend on the
      ``capsys`` or ``capfd`` fixtures. In this case special handling is
      needed to ensure the fixtures take precedence over the global capture.
    """

    def __init__(self, method: _CaptureMethod) -> None:
        self._method: Final = method
        self._global_capturing: Optional[MultiCapture[str]] = None
        self._capture_fixture: Optional[CaptureFixture[Any]] = None

    def __repr__(self) -> str:
        return '<CaptureManager _method={!r} _global_capturing={!r} _capture_fixture={!r}>'.format(self._method, self._global_capturing, self._capture_fixture)

    def is_capturing(self) -> Union[str, bool]:
        if self.is_globally_capturing():
            return 'global'
        if self._capture_fixture:
            return 'fixture %s' % self._capture_fixture.request.fixturename
        return False

    def is_globally_capturing(self) -> bool:
        return self._method != 'no'

    def start_global_capturing(self) -> None:
        assert self._global_capturing is None
        self._global_capturing = _get_multicapture(self._method)
        self._global_capturing.start_capturing()

    def stop_global_capturing(self) -> None:
        if self._global_capturing is not None:
            self._global_capturing.pop_outerr_to_orig()
            self._global_capturing.stop_capturing()
            self._global_capturing = None

    def resume_global_capture(self) -> None:
        if self._global_capturing is not None:
            self._global_capturing.resume_capturing()

    def suspend_global_capture(self, in_: bool=False) -> None:
        if self._global_capturing is not None:
            self._global_capturing.suspend_capturing(in_=in_)

    def suspend(self, in_: bool=False) -> None:
        self.suspend_fixture()
        self.suspend_global_capture(in_)

    def resume(self) -> None:
        self.resume_global_capture()
        self.resume_fixture()

    def read_global_capture(self) -> CaptureResult[str]:
        assert self._global_capturing is not None
        return self._global_capturing.readouterr()

    def set_fixture(self, capture_fixture: 'CaptureFixture[Any]') -> None:
        if self._capture_fixture:
            current_fixture = self._capture_fixture.request.fixturename
            requested_fixture = capture_fixture.request.fixturename
            capture_fixture.request.raiseerror(f'cannot use {requested_fixture} and {current_fixture} at the same time')
        self._capture_fixture = capture_fixture

    def unset_fixture(self) -> None:
        self._capture_fixture = None

    def activate_fixture(self) -> None:
        """If the current item is using ``capsys`` or ``capfd``, activate
        them so they take precedence over the global capture."""
        if self._capture_fixture:
            self._capture_fixture._start()

    def deactivate_fixture(self) -> None:
        """Deactivate the ``capsys`` or ``capfd`` fixture of this item, if any."""
        if self._capture_fixture:
            self._capture_fixture.close()

    def suspend_fixture(self) -> None:
        if self._capture_fixture:
            self._capture_fixture._suspend()

    def resume_fixture(self) -> None:
        if self._capture_fixture:
            self._capture_fixture._resume()

    @contextlib.contextmanager
    def global_and_fixture_disabled(self) -> Generator[None, None, None]:
        """Context manager to temporarily disable global and current fixture capturing."""
        do_fixture = self._capture_fixture and self._capture_fixture._is_started()
        if do_fixture:
            self.suspend_fixture()
        do_global = self._global_capturing and self._global_capturing.is_started()
        if do_global:
            self.suspend_global_capture()
        try:
            yield
        finally:
            if do_global:
                self.resume_global_capture()
            if do_fixture:
                self.resume_fixture()

    @contextlib.contextmanager
    def item_capture(self, when: str, item: Item) -> Generator[None, None, None]:
        self.resume_global_capture()
        self.activate_fixture()
        try:
            yield
        finally:
            self.deactivate_fixture()
            self.suspend_global_capture(in_=False)
            out, err = self.read_global_capture()
            item.add_report_section(when, 'stdout', out)
            item.add_report_section(when, 'stderr', err)

    @hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: Collector) -> Generator[None, CollectReport, CollectReport]:
        if isinstance(collector, File):
            self.resume_global_capture()
            try:
                rep = (yield)
            finally:
                self.suspend_global_capture()
            out, err = self.read_global_capture()
            if out:
                rep.sections.append(('Captured stdout', out))
            if err:
                rep.sections.append(('Captured stderr', err))
        else:
            rep = (yield)
        return rep

    @hookimpl(wrapper=True)
    def pytest_runtest_setup(self, item: Item) -> Generator[None, None, None]:
        with self.item_capture('setup', item):
            return (yield)

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None, None, None]:
        with self.item_capture('call', item):
            return (yield)

    @hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item: Item) -> Generator[None, None, None]:
        with self.item_capture('teardown', item):
            return (yield)

    @hookimpl(tryfirst=True)
    def pytest_keyboard_interrupt(self) -> None:
        self.stop_global_capturing()

    @hookimpl(tryfirst=True)
    def pytest_internalerror(self) -> None:
        self.stop_global_capturing()