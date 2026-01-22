import os
from breezy import tests
from breezy.tests import script
class TestTestScript(tests.TestCaseInTempDir):

    def test_unknnown_file(self):
        self.run_bzr(['test-script', 'I-do-not-exist'], retcode=3)

    def test_empty_file(self):
        self.build_tree_contents([('script', b'')])
        out, err = self.run_bzr(['test-script', 'script'])
        out_lines = out.splitlines()
        self.assertStartsWith(out_lines[-3], 'Ran 1 test in ')
        self.assertEqual('OK', out_lines[-1])
        self.assertEqual('', err)

    def test_simple_file(self):
        self.build_tree_contents([('script', b'\n$ echo hello world\nhello world\n')])
        out, err = self.run_bzr(['test-script', 'script'])
        out_lines = out.splitlines()
        self.assertStartsWith(out_lines[-3], 'Ran 1 test in ')
        self.assertEqual('OK', out_lines[-1])
        self.assertEqual('', err)

    def test_null_output(self):
        self.build_tree_contents([('script', b'\n$ echo hello world\n')])
        out, err = self.run_bzr(['test-script', 'script', '--null-output'])
        out_lines = out.splitlines()
        self.assertStartsWith(out_lines[-3], 'Ran 1 test in ')
        self.assertEqual('OK', out_lines[-1])
        self.assertEqual('', err)

    def test_failing_script(self):
        self.build_tree_contents([('script', b'\n$ echo hello foo\nhello bar\n')])
        out, err = self.run_bzr(['test-script', 'script'], retcode=1)
        out_lines = out.splitlines()
        self.assertStartsWith(out_lines[-3], 'Ran 1 test in ')
        self.assertEqual('FAILED (failures=1)', out_lines[-1])
        self.assertEqual('', err)