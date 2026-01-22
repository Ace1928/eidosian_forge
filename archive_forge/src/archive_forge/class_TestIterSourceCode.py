import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
class TestIterSourceCode(TestCase):
    """
    Tests for L{iterSourceCode}.
    """

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def makeEmptyFile(self, *parts):
        assert parts
        fpath = os.path.join(self.tempdir, *parts)
        open(fpath, 'a').close()
        return fpath

    def test_emptyDirectory(self):
        """
        There are no Python files in an empty directory.
        """
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_singleFile(self):
        """
        If the directory contains one Python file, C{iterSourceCode} will find
        it.
        """
        childpath = self.makeEmptyFile('foo.py')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [childpath])

    def test_onlyPythonSource(self):
        """
        Files that are not Python source files are not included.
        """
        self.makeEmptyFile('foo.pyc')
        self.assertEqual(list(iterSourceCode([self.tempdir])), [])

    def test_recurses(self):
        """
        If the Python files are hidden deep down in child directories, we will
        find them.
        """
        os.mkdir(os.path.join(self.tempdir, 'foo'))
        apath = self.makeEmptyFile('foo', 'a.py')
        self.makeEmptyFile('foo', 'a.py~')
        os.mkdir(os.path.join(self.tempdir, 'bar'))
        bpath = self.makeEmptyFile('bar', 'b.py')
        cpath = self.makeEmptyFile('c.py')
        self.assertEqual(sorted(iterSourceCode([self.tempdir])), sorted([apath, bpath, cpath]))

    def test_shebang(self):
        """
        Find Python files that don't end with `.py`, but contain a Python
        shebang.
        """
        python = os.path.join(self.tempdir, 'a')
        with open(python, 'w') as fd:
            fd.write('#!/usr/bin/env python\n')
        self.makeEmptyFile('b')
        with open(os.path.join(self.tempdir, 'c'), 'w') as fd:
            fd.write('hello\nworld\n')
        python3 = os.path.join(self.tempdir, 'e')
        with open(python3, 'w') as fd:
            fd.write('#!/usr/bin/env python3\n')
        pythonw = os.path.join(self.tempdir, 'f')
        with open(pythonw, 'w') as fd:
            fd.write('#!/usr/bin/env pythonw\n')
        python3args = os.path.join(self.tempdir, 'g')
        with open(python3args, 'w') as fd:
            fd.write('#!/usr/bin/python3 -u\n')
        python3d = os.path.join(self.tempdir, 'i')
        with open(python3d, 'w') as fd:
            fd.write('#!/usr/local/bin/python3d\n')
        python38m = os.path.join(self.tempdir, 'j')
        with open(python38m, 'w') as fd:
            fd.write('#! /usr/bin/env python3.8m\n')
        notfirst = os.path.join(self.tempdir, 'l')
        with open(notfirst, 'w') as fd:
            fd.write('#!/bin/sh\n#!/usr/bin/python\n')
        self.assertEqual(sorted(iterSourceCode([self.tempdir])), sorted([python, python3, pythonw, python3args, python3d, python38m]))

    def test_multipleDirectories(self):
        """
        L{iterSourceCode} can be given multiple directories.  It will recurse
        into each of them.
        """
        foopath = os.path.join(self.tempdir, 'foo')
        barpath = os.path.join(self.tempdir, 'bar')
        os.mkdir(foopath)
        apath = self.makeEmptyFile('foo', 'a.py')
        os.mkdir(barpath)
        bpath = self.makeEmptyFile('bar', 'b.py')
        self.assertEqual(sorted(iterSourceCode([foopath, barpath])), sorted([apath, bpath]))

    def test_explicitFiles(self):
        """
        If one of the paths given to L{iterSourceCode} is not a directory but
        a file, it will include that in its output.
        """
        epath = self.makeEmptyFile('e.py')
        self.assertEqual(list(iterSourceCode([epath])), [epath])