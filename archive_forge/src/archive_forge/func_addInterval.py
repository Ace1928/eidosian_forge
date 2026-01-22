from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def addInterval(self, i):
    if self.intervals:
        delay = self.intervals[0][0] - self.intervals[0][1]
        self.intervals.append([delay + i, i, len(self.intervals)])
        self.intervals.sort()
    else:
        self.intervals.append([i, i, 0])