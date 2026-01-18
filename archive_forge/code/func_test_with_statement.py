import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
def test_with_statement(self):
    fd, path = mkstemp()
    f = os.fdopen(fd, 'w')
    try:
        f.write('foo\nbar\n')
        f.seek(0)
        f.close()
        d = {'path': path}
        suite = Suite('from __future__ import with_statement\nlines = []\nwith open(path) as file:\n    for line in file:\n        lines.append(line)\n')
        suite.execute(d)
        self.assertEqual(['foo\n', 'bar\n'], d['lines'])
    finally:
        os.remove(path)