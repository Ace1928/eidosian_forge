import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_IP32_5k(self):
    return self.__machine(32) and self._is_r5000()