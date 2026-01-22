import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_sparc(self):
    return self.info['isainfo_n'] == 'sparc'