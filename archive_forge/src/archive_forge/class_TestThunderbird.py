from .. import errors, mail_client, osutils, tests, urlutils
class TestThunderbird(tests.TestCase):

    def test_commandline(self):
        tbird = mail_client.Thunderbird(None)
        commandline = tbird._get_compose_commandline(None, None, 'file%')
        self.assertEqual(['-compose', "attachment='%s'" % urlutils.local_path_to_url('file%')], commandline)
        commandline = tbird._get_compose_commandline('jrandom@example.org', 'Hi there!', None, "bo'dy")
        self.assertEqual(['-compose', "body=bo%27dy,subject='Hi there!',to='jrandom@example.org'"], commandline)

    def test_commandline_is_8bit(self):
        tbird = mail_client.Thunderbird(None)
        cmdline = tbird._get_compose_commandline('jrandom@example.org', 'Hi there!', 'file%')
        self.assertEqual(['-compose', "attachment='%s'," % urlutils.local_path_to_url('file%') + "subject='Hi there!',to='jrandom@example.org'"], cmdline)
        for item in cmdline:
            self.assertTrue(isinstance(item, str), 'Command-line item %r is not a native string!' % item)