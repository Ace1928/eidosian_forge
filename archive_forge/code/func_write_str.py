from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import bytes, chr, dict, int, range, super
import re
import io
from string import ascii_letters, digits, hexdigits
def write_str(self, s):
    """Add string s to the accumulated body."""
    self.write(s)
    self.room -= len(s)