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
@staticmethod
def _le_butt(annot, p1, p2, lr, fill_color):
    """Make stream commands for butt line end symbol. "lr" denotes left (False) or right point.
        """
    m, im, L, R, w, scol, fcol, opacity = TOOLS._le_annot_parms(annot, p1, p2, fill_color)
    shift = 3
    d = shift * max(1, w)
    M = R if lr else L
    top = (M + (0, -d / 2.0)) * im
    bot = (M + (0, d / 2.0)) * im
    ap = '\nq\n%s%f %f m\n' % (opacity, top.x, top.y)
    ap += '%f %f l\n' % (bot.x, bot.y)
    ap += '%g w\n' % w
    ap += scol + 's\nQ\n'
    return ap