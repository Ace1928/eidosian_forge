import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def ver_flags(f):
    tokens = f.split('+')
    ver = float('0' + ''.join(re.findall(self._cc_normalize_arch_ver, tokens[0])))
    return (ver, tokens[0], tokens[1:])