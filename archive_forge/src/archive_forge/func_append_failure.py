from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def append_failure(self, report: TestReport) -> None:
    if hasattr(report, 'wasxfail'):
        self._add_simple('skipped', 'xfail-marked test passes unexpectedly')
    else:
        assert report.longrepr is not None
        reprcrash: Optional[ReprFileLocation] = getattr(report.longrepr, 'reprcrash', None)
        if reprcrash is not None:
            message = reprcrash.message
        else:
            message = str(report.longrepr)
        message = bin_xml_escape(message)
        self._add_simple('failure', message, str(report.longrepr))