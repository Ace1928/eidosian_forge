from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_output_option(self):
    stdout = self.run_bzr('send -f branch --output file1')[0]
    self.assertEqual('', stdout)
    md_file = open('file1', 'rb')
    self.addCleanup(md_file.close)
    self.assertContainsRe(md_file.read(), b'rev3')
    stdout = self.run_bzr('send -f branch --output -')[0]
    self.assertContainsRe(stdout, 'rev3')