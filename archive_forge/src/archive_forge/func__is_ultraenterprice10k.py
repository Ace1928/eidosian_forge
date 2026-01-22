import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_ultraenterprice10k(self):
    return re.match('.*Ultra-Enterprise-10000', self.info['uname_i']) is not None