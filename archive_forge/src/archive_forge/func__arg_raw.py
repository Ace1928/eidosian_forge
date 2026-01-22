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
def _arg_raw(dvi, delta):
    """Return *delta* without reading anything more from the dvi file."""
    return delta