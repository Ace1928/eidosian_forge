import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
def _can_target(cmd, arch):
    """Return true if the architecture supports the -arch flag"""
    newcmd = cmd[:]
    fid, filename = tempfile.mkstemp(suffix='.f')
    os.close(fid)
    try:
        d = os.path.dirname(filename)
        output = os.path.splitext(filename)[0] + '.o'
        try:
            newcmd.extend(['-arch', arch, '-c', filename])
            p = Popen(newcmd, stderr=STDOUT, stdout=PIPE, cwd=d)
            p.communicate()
            return p.returncode == 0
        finally:
            if os.path.exists(output):
                os.remove(output)
    finally:
        os.remove(filename)