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
class CreatedUpdateInSrcTests(TestCase):

    def setUp(self):
        self.srcdir = FilePath(self.mktemp())
        self.srcdir.makedirs()
        self.srcdir.child('src').makedirs()
        packagedir = self.srcdir.child('src').child('inctestpkg')
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
        `incremental.update package --path=<path> --dev` increments the dev
        version of the package on the given path
        """
        out = []
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertTrue(self.packagedir.child('_version.py').exists())
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, dev=0)\n__all__ = ["__version__"]\n')
        _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=False, post=False, dev=True, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
        self.assertTrue(self.packagedir.child('_version.py').exists())
        self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 3, dev=1)\n__all__ = ["__version__"]\n')