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
def _handle_tags(self, argv, tagstr):
    found = None
    for x in argv:
        if tagstr in x:
            if found is None:
                found = x
            else:
                raise ValueError('argument %s supplied repeatedly' % tagstr)
    if found is not None:
        posn = argv.index(found)
        try:
            if found == tagstr:
                tag_args = argv[posn + 1].strip()
                argv.remove(tag_args)
            elif '=' in found:
                tag_args = found.split('=')[1].strip()
            else:
                raise AssertionError('unreachable')
        except IndexError:
            msg = '%s requires at least one tag to be specified'
            raise ValueError(msg % tagstr)
        if tag_args.startswith('-'):
            raise ValueError("tag starts with '-', probably a syntax error")
        if '=' in tag_args:
            msg = "%s argument contains '=', probably a syntax error"
            raise ValueError(msg % tagstr)
        attr = tagstr[2:].replace('-', '_')
        setattr(self, attr, tag_args)
        argv.remove(found)