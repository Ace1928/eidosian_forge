import os
import re
import sys
import time
import codecs
import locale
import select
import struct
import platform
import warnings
import functools
import contextlib
import collections
from .color import COLOR_DISTANCE_ALGORITHMS
from .keyboard import (_time_left,
from .sequences import Termcap, Sequence, SequenceTextWrapper
from .colorspace import RGB_256TABLE
from .formatters import (COLORS,
from ._capabilities import CAPABILITY_DATABASE, CAPABILITIES_ADDITIVES, CAPABILITIES_RAW_MIXIN
def __init__color_capabilities(self):
    self._color_distance_algorithm = 'cie2000'
    if not self.does_styling:
        self.number_of_colors = 0
    elif IS_WINDOWS or os.environ.get('COLORTERM') in ('truecolor', '24bit'):
        self.number_of_colors = 1 << 24
    else:
        self.number_of_colors = max(0, curses.tigetnum('colors') or -1)