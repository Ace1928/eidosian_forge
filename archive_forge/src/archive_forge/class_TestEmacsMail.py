from .. import errors, mail_client, osutils, tests, urlutils
class TestEmacsMail(tests.TestCase):

    def test_commandline(self):
        eclient = mail_client.EmacsMail(None)
        commandline = eclient._get_compose_commandline(None, 'Hi there!', None)
        self.assertEqual(['--eval', '(compose-mail nil "Hi there!")'], commandline)
        commandline = eclient._get_compose_commandline('jrandom@example.org', 'Hi there!', None)
        self.assertEqual(['--eval', '(compose-mail "jrandom@example.org" "Hi there!")'], commandline)
        cmdline = eclient._get_compose_commandline(None, None, 'file%')
        if eclient.elisp_tmp_file is not None:
            self.addCleanup(osutils.delete_any, eclient.elisp_tmp_file)
        commandline = ' '.join(cmdline)
        self.assertContainsRe(commandline, '--eval')
        self.assertContainsRe(commandline, '(compose-mail nil nil)')
        self.assertContainsRe(commandline, '(load .*)')
        self.assertContainsRe(commandline, '(bzr-add-mime-att "file%")')

    def test_commandline_is_8bit(self):
        eclient = mail_client.EmacsMail(None)
        commandline = eclient._get_compose_commandline('jrandom@example.org', 'Hi there!', 'file%')
        if eclient.elisp_tmp_file is not None:
            self.addCleanup(osutils.delete_any, eclient.elisp_tmp_file)
        for item in commandline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)