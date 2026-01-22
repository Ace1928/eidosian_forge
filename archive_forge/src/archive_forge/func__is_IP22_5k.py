import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _is_IP22_5k(self):
    return self.__machine(22) and self._is_r5000()