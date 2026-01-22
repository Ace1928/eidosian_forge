import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def are_neighbors(r1, r2):
    """Detect whether r1, r2 are "neighbors".

            Items r1, r2 are called neighbors if the minimum distance between
            their points is less-equal delta.

            Both parameters must be (potentially invalid) rectangles.
            """
    rr1_x0, rr1_x1 = (r1.x0, r1.x1) if r1.x1 > r1.x0 else (r1.x1, r1.x0)
    rr1_y0, rr1_y1 = (r1.y0, r1.y1) if r1.y1 > r1.y0 else (r1.y1, r1.y0)
    rr2_x0, rr2_x1 = (r2.x0, r2.x1) if r2.x1 > r2.x0 else (r2.x1, r2.x0)
    rr2_y0, rr2_y1 = (r2.y0, r2.y1) if r2.y1 > r2.y0 else (r2.y1, r2.y0)
    if 0 or rr1_x1 < rr2_x0 - delta_x or rr1_x0 > rr2_x1 + delta_x or (rr1_y1 < rr2_y0 - delta_y) or (rr1_y0 > rr2_y1 + delta_y):
        return False
    else:
        return True