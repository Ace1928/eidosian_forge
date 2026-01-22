import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_EV4(self):
    return self.is_Alpha() and self.info[0]['cpu model'] == 'EV4'