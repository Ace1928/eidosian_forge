import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_singleCPU(self):
    return len(self.info) == 1