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
def __init__capabilities(self):
    self.caps = collections.OrderedDict()
    for name, (attribute, pattern) in CAPABILITIES_ADDITIVES.items():
        self.caps[name] = Termcap(name, pattern, attribute)
    for name, (attribute, kwds) in CAPABILITY_DATABASE.items():
        if self.does_styling:
            cap = getattr(self, attribute)
            if cap:
                self.caps[name] = Termcap.build(name, cap, attribute, **kwds)
                continue
        pattern = CAPABILITIES_RAW_MIXIN.get(name)
        if pattern:
            self.caps[name] = Termcap(name, pattern, attribute)
    self.caps_compiled = re.compile('|'.join((cap.pattern for name, cap in self.caps.items())))
    self._caps_compiled_any = re.compile('|'.join((cap.named_pattern for name, cap in self.caps.items())) + '|(?P<MISMATCH>.)')
    self._caps_unnamed_any = re.compile('|'.join(('({0})'.format(cap.pattern) for name, cap in self.caps.items())) + '|(.)')