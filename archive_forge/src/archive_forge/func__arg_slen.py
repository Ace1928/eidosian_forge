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
def _arg_slen(dvi, delta):
    """
    Read *delta* bytes, returning None if *delta* is zero, and the bytes
    interpreted as a signed integer otherwise.
    """
    if delta == 0:
        return None
    return dvi._arg(delta, True)