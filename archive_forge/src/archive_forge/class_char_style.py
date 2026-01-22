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
class char_style:

    def __init__(self, rhs=None):
        if rhs:
            self.size = rhs.size
            self.flags = rhs.flags
            self.font = rhs.font
            self.color = rhs.color
            self.asc = rhs.asc
            self.desc = rhs.desc
        else:
            self.size = -1
            self.flags = -1
            self.font = ''
            self.color = -1
            self.asc = 0
            self.desc = 0

    def __str__(self):
        return f'{self.size} {self.flags} {self.font} {self.color} {self.asc} {self.desc}'