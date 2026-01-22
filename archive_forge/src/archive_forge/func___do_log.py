import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.util import StaticClass
import re
import time
import struct
import traceback
def __do_log(self, text):
    """
        Writes the given text verbatim into the log file (if any)
        and/or standard input (if the verbose flag is turned on).

        Used internally.

        @type  text: str
        @param text: Text to print.
        """
    if isinstance(text, compat.unicode):
        text = text.encode('cp1252')
    if self.verbose:
        print(text)
    if self.logfile:
        try:
            self.fd.writelines('%s\n' % text)
        except IOError:
            e = sys.exc_info()[1]
            self.__logfile_error(e)