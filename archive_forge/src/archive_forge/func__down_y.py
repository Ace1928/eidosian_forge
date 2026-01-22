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
@_dispatch(min=161, max=165, state=_dvistate.inpage, args=('slen',))
def _down_y(self, new_y):
    if new_y is not None:
        self.y = new_y
    self.v += self.y