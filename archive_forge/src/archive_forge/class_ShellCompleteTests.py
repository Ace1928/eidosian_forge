from breezy.tests import TestCaseWithTransport
class ShellCompleteTests(TestCaseWithTransport):

    def test_list(self):
        out, err = self.run_bzr('shell-complete')
        self.assertEqual('', err)
        self.assertIn('version:show version of brz\n', out)

    def test_specific_command_missing(self):
        out, err = self.run_bzr('shell-complete missing-command', retcode=3)
        self.assertEqual('brz: ERROR: unknown command "missing-command"\n', err)
        self.assertEqual('', out)

    def test_specific_command(self):
        out, err = self.run_bzr('shell-complete shell-complete')
        self.assertEqual('', err)
        self.assertEqual('"(--help -h)"{--help,-h}\n"(--quiet -q)"{--quiet,-q}\n"(--verbose -v)"{--verbose,-v}\n--usage\ncontext?\n'.splitlines(), sorted(out.splitlines()))