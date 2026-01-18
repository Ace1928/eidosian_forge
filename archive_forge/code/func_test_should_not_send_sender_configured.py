from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_should_not_send_sender_configured(self):
    sender, revid = self.get_sender(sender_configured_config)
    self.assertEqual(False, sender.should_send())