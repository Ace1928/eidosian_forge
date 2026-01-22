import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def approx_equal(lhs, rhs, tolerance=1e-06):
    return abs(lhs - rhs) <= tolerance * min(lhs, rhs)