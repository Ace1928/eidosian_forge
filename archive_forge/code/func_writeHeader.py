import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def writeHeader(self):
    """
        This is the start of the code that calls _arguments
        @return: L{None}
        """
    self.file.write(b'#compdef ' + self.cmdName.encode('utf-8') + b'\n\n_arguments -s -A "-*" \\\n')