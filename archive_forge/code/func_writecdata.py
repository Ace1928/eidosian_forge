from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def writecdata(self, string):
    """Writes text in a CDATA section."""
    self._writeraw('<![CDATA[' + string + ']]>')