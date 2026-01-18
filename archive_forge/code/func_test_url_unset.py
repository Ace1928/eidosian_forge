from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_url_unset(self):
    sender, revid = self.get_sender()
    self.assertEqual(sender.url(), sender.branch.base)