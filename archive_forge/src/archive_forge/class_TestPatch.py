from breezy.tests import TestCaseWithTransport
class TestPatch(TestCaseWithTransport):

    def test_patch(self):
        self.run_bzr('init')
        with open('myfile', 'w') as f:
            f.write('hello')
        self.run_bzr('add')
        self.run_bzr('commit -m hello')
        with open('myfile', 'w') as f:
            f.write('goodbye')
        with open('mypatch', 'w') as f:
            f.write(self.run_bzr('diff -p1', retcode=1)[0])
        self.run_bzr('revert')
        self.assertFileEqual('hello', 'myfile')
        self.run_bzr('patch -p1 --silent mypatch')
        self.assertFileEqual('goodbye', 'myfile')

    def test_patch_invalid_strip(self):
        self.run_bzr_error(args='patch --strip=a', error_regexes=['brz: ERROR: invalid value for option -p/--strip: a'])