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
class SymPyDocTestFinder(DocTestFinder):
    """
    A class used to extract the DocTests that are relevant to a given
    object, from its docstring and the docstrings of its contained
    objects.  Doctests can currently be extracted from the following
    object types: modules, functions, classes, methods, staticmethods,
    classmethods, and properties.

    Modified from doctest's version to look harder for code that
    appears comes from a different module. For example, the @vectorize
    decorator makes it look like functions come from multidimensional.py
    even though their code exists elsewhere.
    """

    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        """
        Find tests for the given object and any contained objects, and
        add them to ``tests``.
        """
        if self._verbose:
            print('Finding tests in %s' % name)
        if id(obj) in seen:
            return
        seen[id(obj)] = 1
        if inspect.isclass(obj):
            if obj.__module__.split('.')[0] != 'sympy':
                return
        test = self._get_test(obj, name, module, globs, source_lines)
        if test is not None:
            tests.append(test)
        if not self._recurse:
            return
        if inspect.ismodule(obj):
            for rawname, val in obj.__dict__.items():
                if inspect.isfunction(val) or inspect.isclass(val):
                    if val.__module__ != module.__name__:
                        continue
                    assert self._from_module(module, val), '%s is not in module %s (rawname %s)' % (val, module, rawname)
                    try:
                        valname = '%s.%s' % (name, rawname)
                        self._find(tests, val, valname, module, source_lines, globs, seen)
                    except KeyboardInterrupt:
                        raise
            for valname, val in getattr(obj, '__test__', {}).items():
                if not isinstance(valname, str):
                    raise ValueError('SymPyDocTestFinder.find: __test__ keys must be strings: %r' % (type(valname),))
                if not (inspect.isfunction(val) or inspect.isclass(val) or inspect.ismethod(val) or inspect.ismodule(val) or isinstance(val, str)):
                    raise ValueError('SymPyDocTestFinder.find: __test__ values must be strings, functions, methods, classes, or modules: %r' % (type(val),))
                valname = '%s.__test__.%s' % (name, valname)
                self._find(tests, val, valname, module, source_lines, globs, seen)
        if inspect.isclass(obj):
            for valname, val in obj.__dict__.items():
                if isinstance(val, staticmethod):
                    val = getattr(obj, valname)
                if isinstance(val, classmethod):
                    val = getattr(obj, valname).__func__
                if (inspect.isfunction(unwrap(val)) or inspect.isclass(val) or isinstance(val, property)) and self._from_module(module, val):
                    if isinstance(val, property):
                        if hasattr(val.fget, '__module__'):
                            if val.fget.__module__ != module.__name__:
                                continue
                    elif val.__module__ != module.__name__:
                        continue
                    assert self._from_module(module, val), '%s is not in module %s (valname %s)' % (val, module, valname)
                    valname = '%s.%s' % (name, valname)
                    self._find(tests, val, valname, module, source_lines, globs, seen)

    def _get_test(self, obj, name, module, globs, source_lines):
        """
        Return a DocTest for the given object, if it defines a docstring;
        otherwise, return None.
        """
        lineno = None
        if isinstance(obj, str):
            docstring = obj
            matches = re.findall('line \\d+', name)
            assert len(matches) == 1, "string '%s' does not contain lineno " % name
            lineno = int(matches[0][5:])
        else:
            try:
                if obj.__doc__ is None:
                    docstring = ''
                else:
                    docstring = obj.__doc__
                    if not isinstance(docstring, str):
                        docstring = str(docstring)
            except (TypeError, AttributeError):
                docstring = ''
        if self._exclude_empty and (not docstring):
            return None
        if isinstance(obj, property):
            if obj.fget.__doc__ is None:
                return None
        if lineno is None:
            obj = unwrap(obj)
            if hasattr(obj, 'func_closure') and obj.func_closure is not None:
                tobj = obj.func_closure[0].cell_contents
            elif isinstance(obj, property):
                tobj = obj.fget
            else:
                tobj = obj
            lineno = self._find_lineno(tobj, source_lines)
        if lineno is None:
            return None
        if module is None:
            filename = None
        else:
            filename = getattr(module, '__file__', module.__name__)
            if filename[-4:] in ('.pyc', '.pyo'):
                filename = filename[:-1]
        globs['_doctest_depends_on'] = getattr(obj, '_doctest_depends_on', {})
        return self._parser.get_doctest(docstring, globs, name, filename, lineno)