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
def CheckOrder(small, big):
    """Ensures small is ordered before big."""
    self.assertFalse(small == big, '%r unexpectedly equals %r' % (small, big))
    self.assertTrue(small != big, '%r unexpectedly equals %r' % (small, big))
    self.assertLess(small, big)
    self.assertFalse(big < small, '%r unexpectedly less than %r' % (big, small))
    self.assertLessEqual(small, big)
    self.assertFalse(big <= small, '%r unexpectedly less than or equal to %r' % (big, small))
    self.assertGreater(big, small)
    self.assertFalse(small > big, '%r unexpectedly greater than %r' % (small, big))
    self.assertGreaterEqual(big, small)
    self.assertFalse(small >= big, '%r unexpectedly greater than or equal to %r' % (small, big))