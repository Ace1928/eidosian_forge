import copy
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import get_rocm_path
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
from cupy_backends.cuda.libs import nvrtc
from cupy import _util
class CompileException(Exception):

    def __init__(self, msg, source, name, options, backend='nvrtc'):
        self._msg = msg
        self.source = source
        self.name = name
        self.options = options
        self.backend = backend
        super(CompileException, self).__init__()

    def __reduce__(self):
        return (type(self), (self._msg, self.source, self.name, self.options, self.backend))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.get_message()

    def get_message(self):
        return self._msg

    def dump(self, f):
        lines = self.source.split('\n')
        digits = int(math.floor(math.log10(len(lines)))) + 1
        linum_fmt = '{{:0{}d}} '.format(digits)
        f.write('{} '.format(self.backend.upper()))
        f.write('compilation error: {}\n'.format(self))
        f.write('-----\n')
        f.write('Name: {}\n'.format(self.name))
        f.write('Options: {}\n'.format(' '.join(self.options)))
        f.write('CUDA source:\n')
        for i, line in enumerate(lines):
            f.write(linum_fmt.format(i + 1) + line.rstrip() + '\n')
        f.write('-----\n')
        f.flush()