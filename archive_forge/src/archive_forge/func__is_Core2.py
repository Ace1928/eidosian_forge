import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_Core2(self):
    return self.is_Intel() and self.info[0]['Family'] == 6 and (self.info[0]['Model'] in [15, 16, 17])