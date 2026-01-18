from breezy.tests import TestCaseWithTransport
def test_specific_command_missing(self):
    out, err = self.run_bzr('shell-complete missing-command', retcode=3)
    self.assertEqual('brz: ERROR: unknown command "missing-command"\n', err)
    self.assertEqual('', out)