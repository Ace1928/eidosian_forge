import bdb
import builtins
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex, import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
def _init_checker_class() -> Type['IPDoctestOutputChecker']:
    import doctest
    import re
    from .ipdoctest import IPDoctestOutputChecker

    class LiteralsOutputChecker(IPDoctestOutputChecker):
        _unicode_literal_re = re.compile('(\\W|^)[uU]([rR]?[\\\'\\"])', re.UNICODE)
        _bytes_literal_re = re.compile('(\\W|^)[bB]([rR]?[\\\'\\"])', re.UNICODE)
        _number_re = re.compile('\n            (?P<number>\n              (?P<mantissa>\n                (?P<integer1> [+-]?\\d*)\\.(?P<fraction>\\d+)\n                |\n                (?P<integer2> [+-]?\\d+)\\.\n              )\n              (?:\n                [Ee]\n                (?P<exponent1> [+-]?\\d+)\n              )?\n              |\n              (?P<integer3> [+-]?\\d+)\n              (?:\n                [Ee]\n                (?P<exponent2> [+-]?\\d+)\n              )\n            )\n            ', re.VERBOSE)

        def check_output(self, want: str, got: str, optionflags: int) -> bool:
            if super().check_output(want, got, optionflags):
                return True
            allow_unicode = optionflags & _get_allow_unicode_flag()
            allow_bytes = optionflags & _get_allow_bytes_flag()
            allow_number = optionflags & _get_number_flag()
            if not allow_unicode and (not allow_bytes) and (not allow_number):
                return False

            def remove_prefixes(regex: Pattern[str], txt: str) -> str:
                return re.sub(regex, '\\1\\2', txt)
            if allow_unicode:
                want = remove_prefixes(self._unicode_literal_re, want)
                got = remove_prefixes(self._unicode_literal_re, got)
            if allow_bytes:
                want = remove_prefixes(self._bytes_literal_re, want)
                got = remove_prefixes(self._bytes_literal_re, got)
            if allow_number:
                got = self._remove_unwanted_precision(want, got)
            return super().check_output(want, got, optionflags)

        def _remove_unwanted_precision(self, want: str, got: str) -> str:
            wants = list(self._number_re.finditer(want))
            gots = list(self._number_re.finditer(got))
            if len(wants) != len(gots):
                return got
            offset = 0
            for w, g in zip(wants, gots):
                fraction: Optional[str] = w.group('fraction')
                exponent: Optional[str] = w.group('exponent1')
                if exponent is None:
                    exponent = w.group('exponent2')
                precision = 0 if fraction is None else len(fraction)
                if exponent is not None:
                    precision -= int(exponent)
                if float(w.group()) == approx(float(g.group()), abs=10 ** (-precision)):
                    got = got[:g.start() + offset] + w.group() + got[g.end() + offset:]
                    offset += w.end() - w.start() - (g.end() - g.start())
            return got
    return LiteralsOutputChecker