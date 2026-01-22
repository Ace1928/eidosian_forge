from .. import errors, mail_client, osutils, tests, urlutils
class TestXDGEmail(tests.TestCase):

    def test_commandline(self):
        xdg_email = mail_client.XDGEmail(None)
        self.assertRaises(mail_client.NoMailAddressSpecified, xdg_email._get_compose_commandline, None, None, 'file%')
        commandline = xdg_email._get_compose_commandline('jrandom@example.org', None, 'file%')
        self.assertEqual(['jrandom@example.org', '--attach', 'file%'], commandline)
        commandline = xdg_email._get_compose_commandline('jrandom@example.org', 'Hi there!', None, "bo'dy")
        self.assertEqual(['jrandom@example.org', '--subject', 'Hi there!', '--body', "bo'dy"], commandline)

    def test_commandline_is_8bit(self):
        xdg_email = mail_client.XDGEmail(None)
        cmdline = xdg_email._get_compose_commandline('jrandom@example.org', 'Hi there!', 'file%')
        self.assertEqual(['jrandom@example.org', '--subject', 'Hi there!', '--attach', 'file%'], cmdline)
        for item in cmdline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)