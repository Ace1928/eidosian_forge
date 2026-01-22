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
class SymPyDocTestRunner(DocTestRunner):
    """
    A class used to run DocTest test cases, and accumulate statistics.
    The ``run`` method is used to process a single DocTest case.  It
    returns a tuple ``(f, t)``, where ``t`` is the number of test cases
    tried, and ``f`` is the number of test cases that failed.

    Modified from the doctest version to not reset the sys.displayhook (see
    issue 5140).

    See the docstring of the original DocTestRunner for more information.
    """

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        """
        Run the examples in ``test``, and display the results using the
        writer function ``out``.

        The examples are run in the namespace ``test.globs``.  If
        ``clear_globs`` is true (the default), then this namespace will
        be cleared after the test runs, to help with garbage
        collection.  If you would like to examine the namespace after
        the test completes, then use ``clear_globs=False``.

        ``compileflags`` gives the set of flags that should be used by
        the Python compiler when running the examples.  If not
        specified, then it will default to the set of future-import
        flags that apply to ``globs``.

        The output of each example is checked using
        ``SymPyDocTestRunner.check_output``, and the results are
        formatted by the ``SymPyDocTestRunner.report_*`` methods.
        """
        self.test = test
        for example in test.examples:
            example.want = example.want.replace('```\n', '')
            example.exc_msg = example.exc_msg and example.exc_msg.replace('```\n', '')
        if compileflags is None:
            compileflags = pdoctest._extract_future_flags(test.globs)
        save_stdout = sys.stdout
        if out is None:
            out = save_stdout.write
        sys.stdout = self._fakeout
        save_set_trace = pdb.set_trace
        self.debugger = pdoctest._OutputRedirectingPdb(save_stdout)
        self.debugger.reset()
        pdb.set_trace = self.debugger.set_trace
        self.save_linecache_getlines = pdoctest.linecache.getlines
        linecache.getlines = self.__patched_linecache_getlines
        with raise_on_deprecated():
            try:
                return self.__run(test, compileflags, out)
            finally:
                sys.stdout = save_stdout
                pdb.set_trace = save_set_trace
                linecache.getlines = self.save_linecache_getlines
                if clear_globs:
                    test.globs.clear()