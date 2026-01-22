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
def assertBetween(self, value, minv, maxv, msg=None):
    """Asserts that value is between minv and maxv (inclusive)."""
    if msg is None:
        msg = '"%r" unexpectedly not between "%r" and "%r"' % (value, minv, maxv)
    self.assert_(minv <= value, msg)
    self.assert_(maxv >= value, msg)