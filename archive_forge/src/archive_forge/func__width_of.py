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
def _width_of(self, char):
    """Width of char in dvi units."""
    width = self._tfm.width.get(char, None)
    if width is not None:
        return _mul2012(width, self._scale)
    _log.debug('No width for char %d in font %s.', char, self.texname)
    return 0