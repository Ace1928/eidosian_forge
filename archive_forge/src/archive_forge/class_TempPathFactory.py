import dataclasses
import os
from pathlib import Path
import re
from shutil import rmtree
import tempfile
from typing import Any
from typing import Dict
from typing import final
from typing import Generator
from typing import Literal
from typing import Optional
from typing import Union
from .pathlib import cleanup_dead_symlinks
from .pathlib import LOCK_TIMEOUT
from .pathlib import make_numbered_dir
from .pathlib import make_numbered_dir_with_cleanup
from .pathlib import rm_rf
from _pytest.compat import get_user_id
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.stash import StashKey
@final
@dataclasses.dataclass
class TempPathFactory:
    """Factory for temporary directories under the common base temp directory.

    The base directory can be configured using the ``--basetemp`` option.
    """
    _given_basetemp: Optional[Path]
    _trace: Any
    _basetemp: Optional[Path]
    _retention_count: int
    _retention_policy: RetentionType

    def __init__(self, given_basetemp: Optional[Path], retention_count: int, retention_policy: RetentionType, trace, basetemp: Optional[Path]=None, *, _ispytest: bool=False) -> None:
        check_ispytest(_ispytest)
        if given_basetemp is None:
            self._given_basetemp = None
        else:
            self._given_basetemp = Path(os.path.abspath(str(given_basetemp)))
        self._trace = trace
        self._retention_count = retention_count
        self._retention_policy = retention_policy
        self._basetemp = basetemp

    @classmethod
    def from_config(cls, config: Config, *, _ispytest: bool=False) -> 'TempPathFactory':
        """Create a factory according to pytest configuration.

        :meta private:
        """
        check_ispytest(_ispytest)
        count = int(config.getini('tmp_path_retention_count'))
        if count < 0:
            raise ValueError(f'tmp_path_retention_count must be >= 0. Current input: {count}.')
        policy = config.getini('tmp_path_retention_policy')
        if policy not in ('all', 'failed', 'none'):
            raise ValueError(f'tmp_path_retention_policy must be either all, failed, none. Current input: {policy}.')
        return cls(given_basetemp=config.option.basetemp, trace=config.trace.get('tmpdir'), retention_count=count, retention_policy=policy, _ispytest=True)

    def _ensure_relative_to_basetemp(self, basename: str) -> str:
        basename = os.path.normpath(basename)
        if (self.getbasetemp() / basename).resolve().parent != self.getbasetemp():
            raise ValueError(f'{basename} is not a normalized and relative path')
        return basename

    def mktemp(self, basename: str, numbered: bool=True) -> Path:
        """Create a new temporary directory managed by the factory.

        :param basename:
            Directory base name, must be a relative path.

        :param numbered:
            If ``True``, ensure the directory is unique by adding a numbered
            suffix greater than any existing one: ``basename="foo-"`` and ``numbered=True``
            means that this function will create directories named ``"foo-0"``,
            ``"foo-1"``, ``"foo-2"`` and so on.

        :returns:
            The path to the new directory.
        """
        basename = self._ensure_relative_to_basetemp(basename)
        if not numbered:
            p = self.getbasetemp().joinpath(basename)
            p.mkdir(mode=448)
        else:
            p = make_numbered_dir(root=self.getbasetemp(), prefix=basename, mode=448)
            self._trace('mktemp', p)
        return p

    def getbasetemp(self) -> Path:
        """Return the base temporary directory, creating it if needed.

        :returns:
            The base temporary directory.
        """
        if self._basetemp is not None:
            return self._basetemp
        if self._given_basetemp is not None:
            basetemp = self._given_basetemp
            if basetemp.exists():
                rm_rf(basetemp)
            basetemp.mkdir(mode=448)
            basetemp = basetemp.resolve()
        else:
            from_env = os.environ.get('PYTEST_DEBUG_TEMPROOT')
            temproot = Path(from_env or tempfile.gettempdir()).resolve()
            user = get_user() or 'unknown'
            rootdir = temproot.joinpath(f'pytest-of-{user}')
            try:
                rootdir.mkdir(mode=448, exist_ok=True)
            except OSError:
                rootdir = temproot.joinpath('pytest-of-unknown')
                rootdir.mkdir(mode=448, exist_ok=True)
            uid = get_user_id()
            if uid is not None:
                rootdir_stat = rootdir.stat()
                if rootdir_stat.st_uid != uid:
                    raise OSError(f'The temporary directory {rootdir} is not owned by the current user. Fix this and try again.')
                if rootdir_stat.st_mode & 63 != 0:
                    os.chmod(rootdir, rootdir_stat.st_mode & ~63)
            keep = self._retention_count
            if self._retention_policy == 'none':
                keep = 0
            basetemp = make_numbered_dir_with_cleanup(prefix='pytest-', root=rootdir, keep=keep, lock_timeout=LOCK_TIMEOUT, mode=448)
        assert basetemp is not None, basetemp
        self._basetemp = basetemp
        self._trace('new basetemp', basetemp)
        return basetemp