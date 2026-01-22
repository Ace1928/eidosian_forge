from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
class TestGetTo(TestCaseInTempDir):

    def test_body(self):
        sender, revid = self.get_sender()
        self.assertEqual('At {}\n\n{}'.format(sender.url(), sample_log % revid.decode('utf-8')), sender.body())

    def test_custom_body(self):
        sender, revid = self.get_sender(customized_mail_config)
        self.assertEqual('%s has committed revision 1 at %s.\n\n%s' % (sender.revision.committer, sender.url(), sample_log % revid.decode('utf-8')), sender.body())

    def test_command_line(self):
        sender, revid = self.get_sender()
        self.assertEqual(['mail', '-s', sender.subject(), '-a', 'From: ' + sender.from_address()] + sender.to(), sender._command_line())

    def test_to(self):
        sender, revid = self.get_sender()
        self.assertEqual(['demo@example.com'], sender.to())

    def test_from(self):
        sender, revid = self.get_sender()
        self.assertEqual('Sample <foo@example.com>', sender.from_address())

    def test_from_default(self):
        sender, revid = self.get_sender(unconfigured_config)
        self.assertEqual('Robert <foo@example.com>', sender.from_address())

    def test_should_send(self):
        sender, revid = self.get_sender()
        self.assertEqual(True, sender.should_send())

    def test_should_not_send(self):
        sender, revid = self.get_sender(unconfigured_config)
        self.assertEqual(False, sender.should_send())

    def test_should_not_send_sender_configured(self):
        sender, revid = self.get_sender(sender_configured_config)
        self.assertEqual(False, sender.should_send())

    def test_should_not_send_to_configured(self):
        sender, revid = self.get_sender(to_configured_config)
        self.assertEqual(True, sender.should_send())

    def test_send_to_multiple(self):
        sender, revid = self.get_sender(multiple_to_configured_config)
        self.assertEqual(['Sample <foo@example.com>', 'Other <baz@bar.com>'], sender.to())
        self.assertEqual(['Sample <foo@example.com>', 'Other <baz@bar.com>'], sender._command_line()[-2:])

    def test_url_set(self):
        sender, revid = self.get_sender(with_url_config)
        self.assertEqual(sender.url(), 'http://some.fake/url/')

    def test_public_url_set(self):
        config = b'[DEFAULT]\npublic_branch=http://the.publication/location/\n'
        sender, revid = self.get_sender(config)
        self.assertEqual(sender.url(), 'http://the.publication/location/')

    def test_url_precedence(self):
        config = b'[DEFAULT]\npost_commit_url=http://some.fake/url/\npublic_branch=http://the.publication/location/\n'
        sender, revid = self.get_sender(config)
        self.assertEqual(sender.url(), 'http://some.fake/url/')

    def test_url_unset(self):
        sender, revid = self.get_sender()
        self.assertEqual(sender.url(), sender.branch.base)

    def test_subject(self):
        sender, revid = self.get_sender()
        self.assertEqual('Rev 1: foo bar baz in %s' % sender.branch.base, sender.subject())

    def test_custom_subject(self):
        sender, revid = self.get_sender(customized_mail_config)
        self.assertEqual('[commit] %s' % sender.revision.get_summary(), sender.subject())

    def test_diff_filename(self):
        sender, revid = self.get_sender()
        self.assertEqual('patch-1.diff', sender.diff_filename())

    def test_headers(self):
        sender, revid = self.get_sender()
        self.assertEqual({'X-Cheese': 'to the rescue!'}, sender.extra_headers())

    def get_sender(self, text=sample_config):
        my_config = config.MemoryStack(text)
        self.branch = BzrDir.create_branch_convenience('.')
        tree = self.branch.controldir.open_workingtree()
        revid = tree.commit('foo bar baz\nfuzzy\nwuzzy', allow_pointless=True, timestamp=1, timezone=0, committer='Sample <john@example.com>')
        sender = EmailSender(self.branch, revid, my_config)
        sender._setup_revision_and_revno()
        return (sender, revid)