import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def __get_nbits(self):
    abits = platform.architecture()[0]
    nbits = re.compile('(\\d+)bit').search(abits).group(1)
    return nbits