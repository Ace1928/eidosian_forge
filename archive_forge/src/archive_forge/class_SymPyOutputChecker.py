import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
class SymPyOutputChecker(pdoctest.OutputChecker):
    """
    Compared to the OutputChecker from the stdlib our OutputChecker class
    supports numerical comparison of floats occurring in the output of the
    doctest examples
    """

    def __init__(self):
        got_floats = '(\\d+\\.\\d*|\\.\\d+)'
        want_floats = got_floats + '(\\.{3})?'
        front_sep = '\\s|\\+|\\-|\\*|,'
        back_sep = front_sep + '|j|e'
        fbeg = '^%s(?=%s|$)' % (got_floats, back_sep)
        fmidend = '(?<=%s)%s(?=%s|$)' % (front_sep, got_floats, back_sep)
        self.num_got_rgx = re.compile('(%s|%s)' % (fbeg, fmidend))
        fbeg = '^%s(?=%s|$)' % (want_floats, back_sep)
        fmidend = '(?<=%s)%s(?=%s|$)' % (front_sep, want_floats, back_sep)
        self.num_want_rgx = re.compile('(%s|%s)' % (fbeg, fmidend))

    def check_output(self, want, got, optionflags):
        """
        Return True iff the actual output from an example (`got`)
        matches the expected output (`want`).  These strings are
        always considered to match if they are identical; but
        depending on what option flags the test runner is using,
        several non-exact match types are also possible.  See the
        documentation for `TestRunner` for more information about
        option flags.
        """
        if got == want:
            return True
        matches = self.num_got_rgx.finditer(got)
        numbers_got = [match.group(1) for match in matches]
        matches = self.num_want_rgx.finditer(want)
        numbers_want = [match.group(1) for match in matches]
        if len(numbers_got) != len(numbers_want):
            return False
        if len(numbers_got) > 0:
            nw_ = []
            for ng, nw in zip(numbers_got, numbers_want):
                if '...' in nw:
                    nw_.append(ng)
                    continue
                else:
                    nw_.append(nw)
                if abs(float(ng) - float(nw)) > 1e-05:
                    return False
            got = self.num_got_rgx.sub('%s', got)
            got = got % tuple(nw_)
        if not optionflags & pdoctest.DONT_ACCEPT_BLANKLINE:
            want = re.sub('(?m)^%s\\s*?$' % re.escape(pdoctest.BLANKLINE_MARKER), '', want)
            got = re.sub('(?m)^\\s*?$', '', got)
            if got == want:
                return True
        if optionflags & pdoctest.NORMALIZE_WHITESPACE:
            got = ' '.join(got.split())
            want = ' '.join(want.split())
            if got == want:
                return True
        if optionflags & pdoctest.ELLIPSIS:
            if pdoctest._ellipsis_match(want, got):
                return True
        return False