import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def _locations_mirror(x):
    """Mirrors the points in a list-of-list-of-...-of-list-of-points.
    For example:
    >>> _locations_mirror([[[1, 2], [3, 4]], [5, 6], [7, 8]])
    [[[2, 1], [4, 3]], [6, 5], [8, 7]]

    """
    if hasattr(x, '__iter__'):
        if hasattr(x[0], '__iter__'):
            return list(map(_locations_mirror, x))
        else:
            return list(x[::-1])
    else:
        return x