import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def test_text_mode(self):
    f = atomicfile.AtomicFile('test', mode='wt')
    f.write(b'foo\n')
    f.commit()
    with open('test', 'rb') as f:
        contents = f.read()
    if sys.platform == 'win32':
        self.assertEqual(b'foo\r\n', contents)
    else:
        self.assertEqual(b'foo\n', contents)