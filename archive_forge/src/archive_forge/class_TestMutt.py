from .. import errors, mail_client, osutils, tests, urlutils
class TestMutt(tests.TestCase):

    def test_commandline(self):
        mutt = mail_client.Mutt(None)
        commandline = mutt._get_compose_commandline(None, None, 'file%', body='hello')
        self.assertEqual(['-a', 'file%', '-i'], commandline[:-1])
        commandline = mutt._get_compose_commandline('jrandom@example.org', 'Hi there!', None)
        self.assertEqual(['-s', 'Hi there!', '--', 'jrandom@example.org'], commandline)

    def test_commandline_is_8bit(self):
        mutt = mail_client.Mutt(None)
        cmdline = mutt._get_compose_commandline('jrandom@example.org', 'Hi there!', 'file%')
        self.assertEqual(['-s', 'Hi there!', '-a', 'file%', '--', 'jrandom@example.org'], cmdline)
        for item in cmdline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)