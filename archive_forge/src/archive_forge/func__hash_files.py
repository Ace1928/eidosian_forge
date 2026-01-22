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
def _hash_files(self, filenames):
    h = hashlib.sha1()
    for fn in filenames:
        with open(fn, 'rb') as f:
            while True:
                block = f.read(131072)
                if not block:
                    break
                h.update(block)
    text = base64.b32encode(h.digest())
    text = text.decode('ascii')
    return text.rstrip('=')