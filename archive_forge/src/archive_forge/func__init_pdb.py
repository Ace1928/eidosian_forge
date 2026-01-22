import argparse
import functools
import sys
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import unittest
from _pytest import outcomes
from _pytest._code import ExceptionInfo
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.config.exceptions import UsageError
from _pytest.nodes import Node
from _pytest.reports import BaseReport
@classmethod
def _init_pdb(cls, method, *args, **kwargs):
    """Initialize PDB debugging, dropping any IO capturing."""
    import _pytest.config
    if cls._pluginmanager is None:
        capman: Optional[CaptureManager] = None
    else:
        capman = cls._pluginmanager.getplugin('capturemanager')
    if capman:
        capman.suspend(in_=True)
    if cls._config:
        tw = _pytest.config.create_terminal_writer(cls._config)
        tw.line()
        if cls._recursive_debug == 0:
            header = kwargs.pop('header', None)
            if header is not None:
                tw.sep('>', header)
            else:
                capturing = cls._is_capturing(capman)
                if capturing == 'global':
                    tw.sep('>', f'PDB {method} (IO-capturing turned off)')
                elif capturing:
                    tw.sep('>', f'PDB {method} (IO-capturing turned off for {capturing})')
                else:
                    tw.sep('>', f'PDB {method}')
    _pdb = cls._import_pdb_cls(capman)(**kwargs)
    if cls._pluginmanager:
        cls._pluginmanager.hook.pytest_enter_pdb(config=cls._config, pdb=_pdb)
    return _pdb