from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import bytes, chr, dict, int, range, super
import re
import io
from string import ascii_letters, digits, hexdigits
def write_wrapped(self, s, extra_room=0):
    """Add a soft line break if needed, then write s."""
    if self.room < len(s) + extra_room:
        self.write_soft_break()
    self.write_str(s)