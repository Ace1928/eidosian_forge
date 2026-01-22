from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
def _fnt_def_real(self, k, c, s, d, a, l):
    n = self.file.read(a + l)
    fontname = n[-l:].decode('ascii')
    tfm = _tfmfile(fontname)
    if c != 0 and tfm.checksum != 0 and (c != tfm.checksum):
        raise ValueError('tfm checksum mismatch: %s' % n)
    try:
        vf = _vffile(fontname)
    except FileNotFoundError:
        vf = None
    self.fonts[k] = DviFont(scale=s, tfm=tfm, texname=n, vf=vf)