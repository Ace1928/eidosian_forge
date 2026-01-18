from __future__ import annotations
from twisted.cred import credentials, error
from twisted.cred.checkers import FilePasswordDB
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words import tap
def test_passwd(self) -> None:
    """
        Tests the --passwd command for backwards-compatibility.
        """
    opt = tap.Options()
    opt.parseOptions(['--passwd', self.file.name])
    self._loginTest(opt)