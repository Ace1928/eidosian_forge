import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Athlon64(self):
    return re.match('.*?Athlon\\(tm\\) 64\\b', self.info[0]['model name']) is not None