from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
def _is_palette_needed(self, p):
    for i in range(0, len(p), 3):
        if not i // 3 == p[i] == p[i + 1] == p[i + 2]:
            return True
    return False