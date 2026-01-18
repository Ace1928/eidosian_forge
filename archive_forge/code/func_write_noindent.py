from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def write_noindent(self, string):
    """Writes text without indentation."""
    self._writeraw(escape(string), indent=False)