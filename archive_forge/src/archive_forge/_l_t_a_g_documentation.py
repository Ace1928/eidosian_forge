from fontTools.misc.textTools import bytesjoin, tobytes, safeEval
from . import DefaultTable
import struct
Add 'tag' to the list of langauge tags if not already there.

        Returns the integer index of 'tag' in the list of all tags.
        