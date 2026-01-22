import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def addRow(self, *row):
    """
        Add a row to the table. All items are converted to strings.

        @type    row: tuple
        @keyword row: Each argument is a cell in the table.
        """
    row = [str(item) for item in row]
    len_row = [len(item) for item in row]
    width = self.__width
    len_old = len(width)
    len_new = len(row)
    known = min(len_old, len_new)
    missing = len_new - len_old
    if missing > 0:
        width.extend(len_row[-missing:])
    elif missing < 0:
        len_row.extend([0] * -missing)
    self.__width = [max(width[i], len_row[i]) for i in compat.xrange(len(len_row))]
    self.__cols.append(row)