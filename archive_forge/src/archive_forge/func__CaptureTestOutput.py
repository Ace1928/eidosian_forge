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
def _CaptureTestOutput(stream, filename):
    """Redirect an output stream to a file.

  Args:
    stream: Should be sys.stdout or sys.stderr.
    filename: File where output should be stored.
  """
    assert not _captured_streams.has_key(stream)
    _captured_streams[stream] = CapturedStream(stream, filename)