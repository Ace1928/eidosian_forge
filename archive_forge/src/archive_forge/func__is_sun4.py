import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_sun4(self):
    return self.info['arch'] == 'sun4'