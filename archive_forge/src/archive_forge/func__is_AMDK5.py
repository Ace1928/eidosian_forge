import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AMDK5(self):
    return self.is_AMD() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] in [0, 1, 2, 3])