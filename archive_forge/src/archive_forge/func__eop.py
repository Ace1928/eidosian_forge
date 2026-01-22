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
@_dispatch(140, state=_dvistate.inpage)
def _eop(self, _):
    self.state = _dvistate.outer
    del self.h, self.v, self.w, self.x, self.y, self.z, self.stack