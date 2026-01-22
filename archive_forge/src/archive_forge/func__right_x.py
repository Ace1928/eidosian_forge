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
@_dispatch(min=152, max=156, state=_dvistate.inpage, args=('slen',))
def _right_x(self, new_x):
    if new_x is not None:
        self.x = new_x
    self.h += self.x