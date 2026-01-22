from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def _copy_meta(self, meta):
    """Make a 2-level deep copy of the meta dictionary."""
    self._meta = Dict()
    for key, val in meta.items():
        if isinstance(val, dict):
            val = Dict(val)
        self._meta[key] = val