from .. import errors, mail_client, osutils, tests, urlutils
class TestClaws(tests.TestCase):

    def test_commandline(self):
        claws = mail_client.Claws(None)
        commandline = claws._get_compose_commandline('jrandom@example.org', None, 'file%')
        self.assertEqual(['--compose', 'mailto:jrandom@example.org?', '--attach', 'file%'], commandline)
        commandline = claws._get_compose_commandline('jrandom@example.org', 'Hi there!', None)
        self.assertEqual(['--compose', 'mailto:jrandom@example.org?subject=Hi%20there%21'], commandline)

    def test_commandline_is_8bit(self):
        claws = mail_client.Claws(None)
        cmdline = claws._get_compose_commandline('jrandom@example.org', 'µcosm of fun!', 'file%')
        subject_string = urlutils.quote('µcosm of fun!'.encode(osutils.get_user_encoding(), 'replace'))
        self.assertEqual(['--compose', 'mailto:jrandom@example.org?subject=%s' % subject_string, '--attach', 'file%'], cmdline)
        for item in cmdline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)

    def test_with_from(self):
        claws = mail_client.Claws(None)
        cmdline = claws._get_compose_commandline('jrandom@example.org', None, None, None, 'qrandom@example.com')
        self.assertEqual(['--compose', 'mailto:jrandom@example.org?from=qrandom%40example.com'], cmdline)

    def test_to_required(self):
        claws = mail_client.Claws(None)
        self.assertRaises(mail_client.NoMailAddressSpecified, claws._get_compose_commandline, None, None, 'file%')

    def test_with_body(self):
        claws = mail_client.Claws(None)
        cmdline = claws._get_compose_commandline('jrandom@example.org', None, None, 'This is some body text')
        self.assertEqual(['--compose', 'mailto:jrandom@example.org?body=This%20is%20some%20body%20text'], cmdline)