from .. import errors, mail_client, osutils, tests, urlutils
class TestEvolution(tests.TestCase):

    def test_commandline(self):
        evo = mail_client.Evolution(None)
        commandline = evo._get_compose_commandline(None, None, 'file%')
        self.assertEqual(['mailto:?attach=file%25'], commandline)
        commandline = evo._get_compose_commandline('jrandom@example.org', 'Hi there!', None, 'bo&dy')
        self.assertEqual(['mailto:jrandom@example.org?body=bo%26dy&subject=Hi%20there%21'], commandline)

    def test_commandline_is_8bit(self):
        evo = mail_client.Evolution(None)
        cmdline = evo._get_compose_commandline('jrandom@example.org', 'Hi there!', 'file%')
        self.assertEqual(['mailto:jrandom@example.org?attach=file%25&subject=Hi%20there%21'], cmdline)
        for item in cmdline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)