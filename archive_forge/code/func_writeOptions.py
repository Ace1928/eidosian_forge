import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def writeOptions(self):
    """
        Write out zsh code for each option in this command
        @return: L{None}
        """
    optNames = list(self.allOptionsNameToDefinition.keys())
    optNames.sort()
    for longname in optNames:
        self.writeOpt(longname)