import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_constant_value(self, jitval, ctype, value):
    assert jitval not in self.constants
    self.constants[jitval] = (ctype, value)