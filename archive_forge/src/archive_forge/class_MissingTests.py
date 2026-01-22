from __future__ import division, absolute_import
import sys
import os
import datetime
from twisted.python.filepath import FilePath
from twisted.python.compat import NativeStringIO
from twisted.trial.unittest import TestCase
from incremental.update import _run, run
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
from incremental import Version
class MissingTests(TestCase):

    def setUp(self):
        self.srcdir = FilePath(self.mktemp())
        self.srcdir.makedirs()
        self.srcdir.child('srca').makedirs()
        packagedir = self.srcdir.child('srca').child('inctestpkg')
        packagedir.makedirs()
        packagedir.child('__init__.py').setContent(b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", "NEXT", 0, 0).short()\nnext_released_version = "inctestpkg NEXT"\n')
        packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3)\n__all__ = ["__version__"]\n')
        self.getcwd = lambda: self.srcdir.path
        self.packagedir = packagedir

        class Date(object):
            year = 2016
            month = 8
        self.date = Date()

    def test_path(self):
        """
        `incremental.update package --dev` raises and quits if it can't find
        the package.
        """
        out = []
        with self.assertRaises(ValueError):
            _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)