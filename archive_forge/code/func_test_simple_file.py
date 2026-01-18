import os
from breezy import tests
from breezy.tests import script
def test_simple_file(self):
    self.build_tree_contents([('script', b'\n$ echo hello world\nhello world\n')])
    out, err = self.run_bzr(['test-script', 'script'])
    out_lines = out.splitlines()
    self.assertStartsWith(out_lines[-3], 'Ran 1 test in ')
    self.assertEqual('OK', out_lines[-1])
    self.assertEqual('', err)