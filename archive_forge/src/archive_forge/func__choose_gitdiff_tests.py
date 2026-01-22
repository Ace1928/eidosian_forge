import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
def _choose_gitdiff_tests(tests, *, use_common_ancestor=False):
    try:
        from git import Repo
    except ImportError:
        raise ValueError('gitpython needed for git functionality')
    repo = Repo('.')
    path = os.path.join('numba', 'tests')
    if use_common_ancestor:
        print(f'Git diff by common ancestor')
        target = 'origin/release0.59...HEAD'
    else:
        target = 'origin/release0.59..HEAD'
    gdiff_paths = repo.git.diff(target, path, name_only=True).split()
    gdiff_paths = [os.path.normpath(x) for x in gdiff_paths]
    selected = []
    gdiff_paths = [os.path.join(repo.working_dir, x) for x in gdiff_paths]
    for test in _flatten_suite(tests):
        assert isinstance(test, unittest.TestCase)
        fname = inspect.getsourcefile(test.__class__)
        if fname in gdiff_paths:
            selected.append(test)
    print('Git diff identified %s tests' % len(selected))
    return unittest.TestSuite(selected)