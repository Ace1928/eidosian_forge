from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
class TestWhoami(tests.TestCaseWithTransport):

    def assertWhoAmI(self, expected, *cmd_args, **kwargs):
        out, err = self.run_bzr(('whoami',) + cmd_args, **kwargs)
        self.assertEqual('', err)
        lines = out.splitlines()
        self.assertLength(1, lines)
        if isinstance(expected, bytes):
            expected = expected.decode(kwargs.get('encoding', 'ascii'))
        self.assertEqual(expected, lines[0].rstrip())

    def test_whoami_no_args_no_conf(self):
        out = self.run_bzr('whoami')[0]
        self.assertTrue(len(out) > 0)
        self.assertEqual(1, out.count('@'))

    def test_whoami_email_no_args(self):
        out = self.run_bzr('whoami --email')[0]
        self.assertTrue(len(out) > 0)
        self.assertEqual(1, out.count('@'))

    def test_whoami_email_arg(self):
        out = self.run_bzr("whoami --email 'foo <foo@example.com>'", 3)[0]
        self.assertEqual('', out)

    def set_branch_email(self, b, email):
        b.get_config_stack().set('email', email)

    def test_whoami_branch(self):
        """branch specific user identity works."""
        wt = self.make_branch_and_tree('.')
        b = branch.Branch.open('.')
        self.set_branch_email(b, 'Branch Identity <branch@identi.ty>')
        self.assertWhoAmI('Branch Identity <branch@identi.ty>')
        self.assertWhoAmI('branch@identi.ty', '--email')
        self.overrideEnv('BRZ_EMAIL', 'Different ID <other@environ.ment>')
        self.assertWhoAmI('Different ID <other@environ.ment>')
        self.assertWhoAmI('other@environ.ment', '--email')

    def test_whoami_utf8(self):
        """verify that an identity can be in utf-8."""
        self.run_bzr(['whoami', 'Branch Identity € <branch@identi.ty>'], encoding='utf-8')
        self.assertWhoAmI(b'Branch Identity \xe2\x82\xac <branch@identi.ty>', encoding='utf-8')
        self.assertWhoAmI('branch@identi.ty', '--email')

    def test_whoami_ascii(self):
        """
        verify that whoami doesn't totally break when in utf-8, using an ascii
        encoding.
        """
        wt = self.make_branch_and_tree('.')
        b = branch.Branch.open('.')
        self.set_branch_email(b, 'Branch Identity € <branch@identi.ty>')
        self.assertWhoAmI('Branch Identity ? <branch@identi.ty>', encoding='ascii')
        self.assertWhoAmI('branch@identi.ty', '--email', encoding='ascii')

    def test_warning(self):
        """verify that a warning is displayed if no email is given."""
        self.make_branch_and_tree('.')
        display = self.run_bzr(['whoami', 'Branch Identity'])[1]
        self.assertEqual('"Branch Identity" does not seem to contain an email address.  This is allowed, but not recommended.\n', display)

    def test_whoami_not_set(self):
        """Ensure whoami error if username is not set and not inferred.
        """
        override_whoami(self)
        out, err = self.run_bzr(['whoami'], 3)
        self.assertContainsRe(err, 'Unable to determine your name')

    def test_whoami_directory(self):
        """Test --directory option."""
        wt = self.make_branch_and_tree('subdir')
        self.set_branch_email(wt.branch, 'Branch Identity <branch@identi.ty>')
        self.assertWhoAmI('Branch Identity <branch@identi.ty>', '--directory', 'subdir')
        self.run_bzr(['whoami', '--directory', 'subdir', '--branch', 'Changed Identity <changed@identi.ty>'])
        wt = wt.controldir.open_workingtree()
        c = wt.branch.get_config_stack()
        self.assertEqual('Changed Identity <changed@identi.ty>', c.get('email'))

    def test_whoami_remote_directory(self):
        """Test --directory option with a remote directory."""
        wt = self.make_branch_and_tree('subdir')
        self.set_branch_email(wt.branch, 'Branch Identity <branch@identi.ty>')
        url = self.get_readonly_url() + '/subdir'
        self.assertWhoAmI('Branch Identity <branch@identi.ty>', '--directory', url)
        url = self.get_url('subdir')
        self.run_bzr(['whoami', '--directory', url, '--branch', 'Changed Identity <changed@identi.ty>'])
        c = branch.Branch.open(url).get_config_stack()
        self.assertEqual('Changed Identity <changed@identi.ty>', c.get('email'))
        override_whoami(self)
        global_conf = config.GlobalStack()
        self.assertRaises(errors.NoWhoami, global_conf.get, 'email')

    def test_whoami_nonbranch_directory(self):
        """Test --directory mentioning a non-branch directory."""
        wt = self.build_tree(['subdir/'])
        out, err = self.run_bzr('whoami --directory subdir', retcode=3)
        self.assertContainsRe(err, 'ERROR: Not a branch')