import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def DocFileTest(path, module_relative=True, package=None, globs=None, parser=DocTestParser(), encoding=None, **options):
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    if package and (not module_relative):
        raise ValueError('Package may only be specified for module-relative paths.')
    doc, path = _load_testfile(path, package, module_relative, encoding or 'utf-8')
    if '__file__' not in globs:
        globs['__file__'] = path
    name = os.path.basename(path)
    test = parser.get_doctest(doc, globs, name, path, 0)
    return DocFileCase(test, **options)