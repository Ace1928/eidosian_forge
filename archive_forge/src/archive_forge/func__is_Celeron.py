import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Celeron(self):
    return re.match('.*?Celeron', self.info[0]['model name']) is not None