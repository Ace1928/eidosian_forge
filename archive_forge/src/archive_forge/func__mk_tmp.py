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
def _mk_tmp(request: FixtureRequest, factory: TempPathFactory) -> Path:
    name = request.node.name
    name = re.sub('[\\W]', '_', name)
    MAXVAL = 30
    name = name[:MAXVAL]
    return factory.mktemp(name, numbered=True)