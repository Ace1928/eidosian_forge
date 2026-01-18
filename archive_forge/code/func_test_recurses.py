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