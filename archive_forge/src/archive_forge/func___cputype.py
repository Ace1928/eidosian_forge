import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def __cputype(self, n):
    return self.info.get('PROCESSORS').split()[0].lower() == 'r%s' % n