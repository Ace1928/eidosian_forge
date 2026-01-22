from fontTools.misc.textTools import bytesjoin, tobytes, safeEval
from . import DefaultTable
import struct
def addTag(self, tag):
    """Add 'tag' to the list of langauge tags if not already there.

        Returns the integer index of 'tag' in the list of all tags.
        """
    try:
        return self.tags.index(tag)
    except ValueError:
        self.tags.append(tag)
        return len(self.tags) - 1