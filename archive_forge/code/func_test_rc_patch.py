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
def test_rc_patch(self):
    """
        `incremental.update package --patch --rc` increments the patch
        version and makes it a release candidate.
        """
    out = []
    _run(u'inctestpkg', path=None, newversion=None, patch=True, rc=True, post=False, dev=False, create=False, _date=self.date, _getcwd=self.getcwd, _print=out.append)
    self.assertEqual(self.packagedir.child('_version.py').getContent(), b'"""\nProvides inctestpkg version information.\n"""\n\n# This file is auto-generated! Do not edit!\n# Use `python -m incremental.update inctestpkg` to change this file.\n\nfrom incremental import Version\n\n__version__ = Version("inctestpkg", 1, 2, 4, release_candidate=1)\n__all__ = ["__version__"]\n')
    self.assertEqual(self.packagedir.child('__init__.py').getContent(), b'\nfrom incremental import Version\nintroduced_in = Version("inctestpkg", 1, 2, 4, release_candidate=1).short()\nnext_released_version = "inctestpkg 1.2.4.rc1"\n')