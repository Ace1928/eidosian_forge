import collections.abc
import contextlib
from fnmatch import fnmatch
import gc
import importlib
from io import StringIO
import locale
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary
from iniconfig import IniConfig
from iniconfig import SectionWrapper
from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning
@final
class HookRecorder:
    """Record all hooks called in a plugin manager.

    Hook recorders are created by :class:`Pytester`.

    This wraps all the hook calls in the plugin manager, recording each call
    before propagating the normal calls.
    """

    def __init__(self, pluginmanager: PytestPluginManager, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        self._pluginmanager = pluginmanager
        self.calls: List[RecordedHookCall] = []
        self.ret: Optional[Union[int, ExitCode]] = None

        def before(hook_name: str, hook_impls, kwargs) -> None:
            self.calls.append(RecordedHookCall(hook_name, kwargs))

        def after(outcome, hook_name: str, hook_impls, kwargs) -> None:
            pass
        self._undo_wrapping = pluginmanager.add_hookcall_monitoring(before, after)

    def finish_recording(self) -> None:
        self._undo_wrapping()

    def getcalls(self, names: Union[str, Iterable[str]]) -> List[RecordedHookCall]:
        """Get all recorded calls to hooks with the given names (or name)."""
        if isinstance(names, str):
            names = names.split()
        return [call for call in self.calls if call._name in names]

    def assert_contains(self, entries: Sequence[Tuple[str, str]]) -> None:
        __tracebackhide__ = True
        i = 0
        entries = list(entries)
        backlocals = sys._getframe(1).f_locals
        while entries:
            name, check = entries.pop(0)
            for ind, call in enumerate(self.calls[i:]):
                if call._name == name:
                    print('NAMEMATCH', name, call)
                    if eval(check, backlocals, call.__dict__):
                        print('CHECKERMATCH', repr(check), '->', call)
                    else:
                        print('NOCHECKERMATCH', repr(check), '-', call)
                        continue
                    i += ind + 1
                    break
                print('NONAMEMATCH', name, 'with', call)
            else:
                fail(f'could not find {name!r} check {check!r}')

    def popcall(self, name: str) -> RecordedHookCall:
        __tracebackhide__ = True
        for i, call in enumerate(self.calls):
            if call._name == name:
                del self.calls[i]
                return call
        lines = [f'could not find call {name!r}, in:']
        lines.extend(['  %s' % x for x in self.calls])
        fail('\n'.join(lines))

    def getcall(self, name: str) -> RecordedHookCall:
        values = self.getcalls(name)
        assert len(values) == 1, (name, values)
        return values[0]

    @overload
    def getreports(self, names: "Literal['pytest_collectreport']") -> Sequence[CollectReport]:
        ...

    @overload
    def getreports(self, names: "Literal['pytest_runtest_logreport']") -> Sequence[TestReport]:
        ...

    @overload
    def getreports(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        ...

    def getreports(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        return [x.report for x in self.getcalls(names)]

    def matchreport(self, inamepart: str='', names: Union[str, Iterable[str]]=('pytest_runtest_logreport', 'pytest_collectreport'), when: Optional[str]=None) -> Union[CollectReport, TestReport]:
        """Return a testreport whose dotted import path matches."""
        values = []
        for rep in self.getreports(names=names):
            if not when and rep.when != 'call' and rep.passed:
                continue
            if when and rep.when != when:
                continue
            if not inamepart or inamepart in rep.nodeid.split('::'):
                values.append(rep)
        if not values:
            raise ValueError(f'could not find test report matching {inamepart!r}: no test reports at all!')
        if len(values) > 1:
            raise ValueError(f'found 2 or more testreports matching {inamepart!r}: {values}')
        return values[0]

    @overload
    def getfailures(self, names: "Literal['pytest_collectreport']") -> Sequence[CollectReport]:
        ...

    @overload
    def getfailures(self, names: "Literal['pytest_runtest_logreport']") -> Sequence[TestReport]:
        ...

    @overload
    def getfailures(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        ...

    def getfailures(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        return [rep for rep in self.getreports(names) if rep.failed]

    def getfailedcollections(self) -> Sequence[CollectReport]:
        return self.getfailures('pytest_collectreport')

    def listoutcomes(self) -> Tuple[Sequence[TestReport], Sequence[Union[CollectReport, TestReport]], Sequence[Union[CollectReport, TestReport]]]:
        passed = []
        skipped = []
        failed = []
        for rep in self.getreports(('pytest_collectreport', 'pytest_runtest_logreport')):
            if rep.passed:
                if rep.when == 'call':
                    assert isinstance(rep, TestReport)
                    passed.append(rep)
            elif rep.skipped:
                skipped.append(rep)
            else:
                assert rep.failed, f'Unexpected outcome: {rep!r}'
                failed.append(rep)
        return (passed, skipped, failed)

    def countoutcomes(self) -> List[int]:
        return [len(x) for x in self.listoutcomes()]

    def assertoutcome(self, passed: int=0, skipped: int=0, failed: int=0) -> None:
        __tracebackhide__ = True
        from _pytest.pytester_assertions import assertoutcome
        outcomes = self.listoutcomes()
        assertoutcome(outcomes, passed=passed, skipped=skipped, failed=failed)

    def clear(self) -> None:
        self.calls[:] = []