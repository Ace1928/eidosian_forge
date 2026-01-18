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
def write_captured_output(self, report: TestReport) -> None:
    if not self.xml.log_passing_tests and report.passed:
        return
    content_out = report.capstdout
    content_log = report.caplog
    content_err = report.capstderr
    if self.xml.logging == 'no':
        return
    content_all = ''
    if self.xml.logging in ['log', 'all']:
        content_all = self._prepare_content(content_log, ' Captured Log ')
    if self.xml.logging in ['system-out', 'out-err', 'all']:
        content_all += self._prepare_content(content_out, ' Captured Out ')
        self._write_content(report, content_all, 'system-out')
        content_all = ''
    if self.xml.logging in ['system-err', 'out-err', 'all']:
        content_all += self._prepare_content(content_err, ' Captured Err ')
        self._write_content(report, content_all, 'system-err')
        content_all = ''
    if content_all:
        self._write_content(report, content_all, 'system-out')