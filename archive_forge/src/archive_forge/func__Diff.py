import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def _Diff(lhs, rhs):
    """Run standard unix 'diff' against two files."""
    cmd = '${TEST_DIFF:-diff} %s %s' % (commands.mkarg(lhs), commands.mkarg(rhs))
    status, output = commands.getstatusoutput(cmd)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 1:
        raise OutputDifferedError('\nRunning %s\n%s\nTest output differed from golden file\n' % (cmd, output))
    elif not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise DiffFailureError('\nRunning %s\n%s\nFailure diffing test output with golden file\n' % (cmd, output))