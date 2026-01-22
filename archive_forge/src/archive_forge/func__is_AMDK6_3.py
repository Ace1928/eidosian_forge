import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_AMDK6_3(self):
    return self.is_AMD() and self.info[0]['Family'] == 5 and (self.info[0]['Model'] == 9)