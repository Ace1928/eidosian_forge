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
def _enter_pdb(node: Node, excinfo: ExceptionInfo[BaseException], rep: BaseReport) -> BaseReport:
    tw = node.config.pluginmanager.getplugin('terminalreporter')._tw
    tw.line()
    showcapture = node.config.option.showcapture
    for sectionname, content in (('stdout', rep.capstdout), ('stderr', rep.capstderr), ('log', rep.caplog)):
        if showcapture in (sectionname, 'all') and content:
            tw.sep('>', 'captured ' + sectionname)
            if content[-1:] == '\n':
                content = content[:-1]
            tw.line(content)
    tw.sep('>', 'traceback')
    rep.toterminal(tw)
    tw.sep('>', 'entering PDB')
    tb = _postmortem_traceback(excinfo)
    rep._pdbshown = True
    post_mortem(tb)
    return rep