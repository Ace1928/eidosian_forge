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
def _height_depth_of(self, char):
    """Height and depth of char in dvi units."""
    result = []
    for metric, name in ((self._tfm.height, 'height'), (self._tfm.depth, 'depth')):
        value = metric.get(char, None)
        if value is None:
            _log.debug('No %s for char %d in font %s', name, char, self.texname)
            result.append(0)
        else:
            result.append(_mul2012(value, self._scale))
    if re.match(b'^cmsy\\d+$', self.texname) and char == 0:
        result[-1] = 0
    return result