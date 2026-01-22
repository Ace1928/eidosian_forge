import dataclasses
from pathlib import Path
import shlex
import subprocess
from typing import Final
from typing import final
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from iniconfig import SectionWrapper
from _pytest.cacheprovider import Cache
from _pytest.compat import LEGACY_PATH
from _pytest.compat import legacy_path
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.pytester import HookRecorder
from _pytest.pytester import Pytester
from _pytest.pytester import RunResult
from _pytest.terminal import TerminalReporter
from _pytest.tmpdir import TempPathFactory
class LegacyTmpdirPlugin:

    @staticmethod
    @fixture(scope='session')
    def tmpdir_factory(request: FixtureRequest) -> TempdirFactory:
        """Return a :class:`pytest.TempdirFactory` instance for the test session."""
        return request.config._tmpdirhandler

    @staticmethod
    @fixture
    def tmpdir(tmp_path: Path) -> LEGACY_PATH:
        """Return a temporary directory path object which is unique to each test
        function invocation, created as a sub directory of the base temporary
        directory.

        By default, a new base temporary directory is created each test session,
        and old bases are removed after 3 sessions, to aid in debugging. If
        ``--basetemp`` is used then it is cleared each session. See
        :ref:`temporary directory location and retention`.

        The returned object is a `legacy_path`_ object.

        .. note::
            These days, it is preferred to use ``tmp_path``.

            :ref:`About the tmpdir and tmpdir_factory fixtures<tmpdir and tmpdir_factory>`.

        .. _legacy_path: https://py.readthedocs.io/en/latest/path.html
        """
        return legacy_path(tmp_path)