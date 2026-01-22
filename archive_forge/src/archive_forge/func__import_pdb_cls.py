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
def _import_pdb_cls(cls, capman: Optional['CaptureManager']):
    if not cls._config:
        import pdb
        return pdb.Pdb
    usepdb_cls = cls._config.getvalue('usepdb_cls')
    if cls._wrapped_pdb_cls and cls._wrapped_pdb_cls[0] == usepdb_cls:
        return cls._wrapped_pdb_cls[1]
    if usepdb_cls:
        modname, classname = usepdb_cls
        try:
            __import__(modname)
            mod = sys.modules[modname]
            parts = classname.split('.')
            pdb_cls = getattr(mod, parts[0])
            for part in parts[1:]:
                pdb_cls = getattr(pdb_cls, part)
        except Exception as exc:
            value = ':'.join((modname, classname))
            raise UsageError(f'--pdbcls: could not import {value!r}: {exc}') from exc
    else:
        import pdb
        pdb_cls = pdb.Pdb
    wrapped_cls = cls._get_pdb_wrapper_class(pdb_cls, capman)
    cls._wrapped_pdb_cls = (usepdb_cls, wrapped_cls)
    return wrapped_cls