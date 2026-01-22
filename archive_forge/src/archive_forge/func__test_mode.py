import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def _test_mode(self, mode):
    if not self.can_sys_preserve_mode():
        raise TestSkipped('This test cannot be run on your platform')
    f = atomicfile.AtomicFile('test', mode='wb', new_mode=mode)
    f.write(b'foo\n')
    f.commit()
    st = os.lstat('test')
    self.assertEqualMode(mode, stat.S_IMODE(st.st_mode))