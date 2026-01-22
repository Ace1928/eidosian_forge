import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import weakref
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook
from matplotlib.testing.exceptions import ImageComparisonFailure
class _GSConverter(_Converter):

    def __call__(self, orig, dest):
        if not self._proc:
            self._proc = subprocess.Popen([mpl._get_executable_info('gs').executable, '-dNOSAFER', '-dNOPAUSE', '-dEPSCrop', '-sDEVICE=png16m'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                self._read_until(b'\nGS')
            except _ConverterError as e:
                raise OSError(f'Failed to start Ghostscript:\n\n{e.args[0]}') from None

        def encode_and_escape(name):
            return os.fsencode(name).replace(b'\\', b'\\\\').replace(b'(', b'\\(').replace(b')', b'\\)')
        self._proc.stdin.write(b'<< /OutputFile (' + encode_and_escape(dest) + b') >> setpagedevice (' + encode_and_escape(orig) + b') run flush\n')
        self._proc.stdin.flush()
        err = self._read_until((b'GS<', b'GS>'))
        stack = self._read_until(b'>') if err.endswith(b'GS<') else b''
        if stack or not os.path.exists(dest):
            stack_size = int(stack[:-1]) if stack else 0
            self._proc.stdin.write(b'pop\n' * stack_size)
            raise ImageComparisonFailure((err + stack).decode(sys.getfilesystemencoding(), 'replace'))