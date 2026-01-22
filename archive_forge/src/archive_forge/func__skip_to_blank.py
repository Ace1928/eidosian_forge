import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def _skip_to_blank(f, spacegroup, setting):
    """Read lines from f until a blank line is encountered."""
    while True:
        line = f.readline()
        if not line:
            raise SpacegroupNotFoundError('invalid spacegroup `%s`, setting `%s` not found in data base' % (spacegroup, setting))
        if not line.strip():
            break