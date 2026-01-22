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
def appendv(self, pos, text, font=None, fontsize=11, language=None, small_caps=False):
    lheight = fontsize * 1.2
    for c in text:
        self.append(pos, c, font=font, fontsize=fontsize, language=language, small_caps=small_caps)
        pos.y += lheight
    return (self.text_rect, self.last_point)