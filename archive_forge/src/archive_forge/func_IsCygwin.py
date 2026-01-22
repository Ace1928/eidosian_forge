import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def IsCygwin():
    try:
        out = subprocess.Popen('uname', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = out.communicate()[0].decode('utf-8')
        return 'CYGWIN' in str(stdout)
    except Exception:
        return False