from __future__ import annotations
import configparser
import importlib.metadata
import inspect
import itertools
import logging
import sys
from typing import Any
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from flake8 import utils
from flake8.defaults import VALID_CODE_PREFIX
from flake8.exceptions import ExecutionError
from flake8.exceptions import FailedToLoadPlugin
def _flake8_plugins(eps: Iterable[importlib.metadata.EntryPoint], name: str, version: str) -> Generator[Plugin, None, None]:
    pyflakes_meta = importlib.metadata.distribution('pyflakes').metadata
    pycodestyle_meta = importlib.metadata.distribution('pycodestyle').metadata
    for ep in eps:
        if ep.group not in FLAKE8_GROUPS:
            continue
        if ep.name == 'F':
            yield Plugin(pyflakes_meta['name'], pyflakes_meta['version'], ep)
        elif ep.name in 'EW':
            yield Plugin(pycodestyle_meta['name'], pycodestyle_meta['version'], ep)
        else:
            yield Plugin(name, version, ep)