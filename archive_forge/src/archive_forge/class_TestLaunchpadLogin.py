from ...tests import TestCaseWithTransport
from . import account
class TestLaunchpadLogin(TestCaseWithTransport):
    """Tests for launchpad-login."""

    def test_login_without_name_when_not_logged_in(self):
        out, err = self.run_bzr(['launchpad-login', '--no-check'], retcode=1)
        self.assertEqual('No Launchpad user ID configured.\n', out)
        self.assertEqual('', err)

    def test_login_with_name_sets_login(self):
        self.run_bzr(['launchpad-login', '--no-check', 'foo'])
        self.assertEqual('foo', account.get_lp_login())

    def test_login_without_name_when_logged_in(self):
        account.set_lp_login('foo')
        out, err = self.run_bzr(['launchpad-login', '--no-check'])
        self.assertEqual('foo\n', out)
        self.assertEqual('', err)

    def test_login_with_name_no_output_by_default(self):
        out, err = self.run_bzr(['launchpad-login', '--no-check', 'foo'])
        self.assertEqual('', out)
        self.assertEqual('', err)

    def test_login_with_name_verbose(self):
        out, err = self.run_bzr(['launchpad-login', '-v', '--no-check', 'foo'])
        self.assertEqual("Launchpad user ID set to 'foo'.\n", out)
        self.assertEqual('', err)

    def test_logout(self):
        out, err = self.run_bzr(['launchpad-login', '-v', '--no-check', 'foo'])
        self.assertEqual("Launchpad user ID set to 'foo'.\n", out)
        self.assertEqual('', err)
        out, err = self.run_bzr(['launchpad-logout', '-v'])
        self.assertEqual('Launchpad user ID foo logged out.\n', out)
        self.assertEqual('', err)

    def test_logout_not_logged_in(self):
        out, err = self.run_bzr(['launchpad-logout', '-v'], retcode=1)
        self.assertEqual('Not logged into Launchpad.\n', out)
        self.assertEqual('', err)