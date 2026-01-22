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
def _doctest(*paths, **kwargs):
    """
    Internal function that actually runs the doctests.

    All keyword arguments from ``doctest()`` are passed to this function
    except for ``subprocess``.

    Returns 0 if tests passed and 1 if they failed.  See the docstrings of
    ``doctest()`` and ``test()`` for more information.
    """
    from sympy.printing.pretty.pretty import pprint_use_unicode
    normal = kwargs.get('normal', False)
    verbose = kwargs.get('verbose', False)
    colors = kwargs.get('colors', True)
    force_colors = kwargs.get('force_colors', False)
    blacklist = kwargs.get('blacklist', [])
    split = kwargs.get('split', None)
    blacklist.extend(_get_doctest_blacklist())
    if import_module('matplotlib') is not None:
        import matplotlib
        matplotlib.use('Agg')
    import sympy.external
    sympy.external.importtools.WARN_OLD_VERSION = False
    sympy.external.importtools.WARN_NOT_INSTALLED = False
    from sympy.plotting.plot import unset_show
    unset_show()
    r = PyTestReporter(verbose, split=split, colors=colors, force_colors=force_colors)
    t = SymPyDocTests(r, normal)
    test_files = t.get_test_files('sympy')
    test_files.extend(t.get_test_files('examples', init_only=False))
    not_blacklisted = [f for f in test_files if not any((b in f for b in blacklist))]
    if len(paths) == 0:
        matched = not_blacklisted
    else:
        paths = convert_to_native_paths(paths)
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break
    matched.sort()
    if split:
        matched = split_list(matched, split)
    t._testfiles.extend(matched)
    if t._testfiles:
        failed = not t.test()
    else:
        failed = False
    test_files_rst = t.get_test_files('doc/src', '*.rst', init_only=False)
    test_files_md = t.get_test_files('doc/src', '*.md', init_only=False)
    test_files = test_files_rst + test_files_md
    test_files.sort()
    not_blacklisted = [f for f in test_files if not any((b in f for b in blacklist))]
    if len(paths) == 0:
        matched = not_blacklisted
    else:
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break
    if split:
        matched = split_list(matched, split)
    first_report = True
    for rst_file in matched:
        if not os.path.isfile(rst_file):
            continue
        old_displayhook = sys.displayhook
        try:
            use_unicode_prev = setup_pprint()
            out = sympytestfile(rst_file, module_relative=False, encoding='utf-8', optionflags=pdoctest.ELLIPSIS | pdoctest.NORMALIZE_WHITESPACE | pdoctest.IGNORE_EXCEPTION_DETAIL)
        finally:
            sys.displayhook = old_displayhook
            import sympy.interactive.printing as interactive_printing
            interactive_printing.NO_GLOBAL = False
            pprint_use_unicode(use_unicode_prev)
        rstfailed, tested = out
        if tested:
            failed = rstfailed or failed
            if first_report:
                first_report = False
                msg = 'rst/md doctests start'
                if not t._testfiles:
                    r.start(msg=msg)
                else:
                    r.write_center(msg)
                    print()
            file_id = rst_file[rst_file.find('sympy') + len('sympy') + 1:]
            print(file_id, end=' ')
            wid = r.terminal_width - len(file_id) - 1
            test_file = '[%s]' % tested
            report = '[%s]' % (rstfailed or 'OK')
            print(''.join([test_file, ' ' * (wid - len(test_file) - len(report)), report]))
    if not first_report and failed:
        print()
        print('DO *NOT* COMMIT!')
    return int(failed)