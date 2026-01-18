from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_public_url_set(self):
    config = b'[DEFAULT]\npublic_branch=http://the.publication/location/\n'
    sender, revid = self.get_sender(config)
    self.assertEqual(sender.url(), 'http://the.publication/location/')