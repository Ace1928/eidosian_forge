import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AthlonHX(self):
    return re.match('.*?Athlon HX\\b', self.info[0]['model name']) is not None