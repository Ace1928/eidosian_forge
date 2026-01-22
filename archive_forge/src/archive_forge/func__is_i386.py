import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_i386(self):
    return self.info[0]['Family'] == 3