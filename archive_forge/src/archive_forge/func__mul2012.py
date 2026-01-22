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
def _mul2012(num1, num2):
    """Multiply two numbers in 20.12 fixed point format."""
    return num1 * num2 >> 20