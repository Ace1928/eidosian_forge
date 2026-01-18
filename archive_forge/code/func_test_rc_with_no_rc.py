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
def test_rc_with_no_rc(self):
    """
        `incremental.update package --rc`, when the package is not a release
        candidate, will issue a new major/minor rc, and disregards the micro
        and dev.
        """
    self.packagedir.child('_version.py').setContent(b'\nfrom incremental import Version\n__version__ = Version("inctestpkg", 1, 2, 3, dev=2)\n__all__ = ["__version__"]\n')
    out = []
    _run(u'inctestpkg', path=None, newversion=None, patch=False, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 16, 8, 0, release_candidate=1)\n__all__ = ["__version__"]\n')
    self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 16, 8, 0, release_candidate=1).short()\nnext_released_version = "inctestpkg 16.8.0.rc1"\n')