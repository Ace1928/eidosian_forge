import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def yieldOutput(self):
    """
        Generate the text output for the table.

        @rtype:  generator of str
        @return: Text output.
        """
    width = self.__width
    if width:
        num_cols = len(width)
        fmt = ['%%%ds' % -w for w in width]
        if width[-1] > 0:
            fmt[-1] = '%s'
        fmt = self.__sep.join(fmt)
        for row in self.__cols:
            row.extend([''] * (num_cols - len(row)))
            yield (fmt % tuple(row))