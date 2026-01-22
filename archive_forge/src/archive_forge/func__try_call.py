import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def _try_call(self, func):
    try:
        return func()
    except Exception:
        pass