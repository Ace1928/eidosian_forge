import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_subprocess_call_bad_file(self):
    if sys.platform != 'win32':
        raise TestNotApplicable('Workarounds for windows only')
    import errno
    import subprocess
    ERROR_BAD_EXE_FORMAT = 193
    open('textfile.txt', 'w').close()
    e = self.assertRaises(WindowsError, subprocess.call, 'textfile.txt')
    self.assertEqual(e.errno, errno.ENOEXEC)
    self.assertEqual(e.winerror, ERROR_BAD_EXE_FORMAT)