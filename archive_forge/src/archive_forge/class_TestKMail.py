from .. import errors, mail_client, osutils, tests, urlutils
class TestKMail(tests.TestCase):

    def test_commandline(self):
        kmail = mail_client.KMail(None)
        commandline = kmail._get_compose_commandline(None, None, 'file%')
        self.assertEqual(['--attach', 'file%'], commandline)
        commandline = kmail._get_compose_commandline('jrandom@example.org', 'Hi there!', None)
        self.assertEqual(['-s', 'Hi there!', 'jrandom@example.org'], commandline)

    def test_commandline_is_8bit(self):
        kmail = mail_client.KMail(None)
        cmdline = kmail._get_compose_commandline('jrandom@example.org', 'Hi there!', 'file%')
        self.assertEqual(['-s', 'Hi there!', '--attach', 'file%', 'jrandom@example.org'], cmdline)
        for item in cmdline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)