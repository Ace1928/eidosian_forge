from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_subject(self):
    sender, revid = self.get_sender()
    self.assertEqual('Rev 1: foo bar baz in %s' % sender.branch.base, sender.subject())