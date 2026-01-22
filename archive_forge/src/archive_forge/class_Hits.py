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
class Hits:

    def __str__(self):
        return f'Hits(len={self.len} quads={self.quads} hfuzz={self.hfuzz} vfuzz={self.vfuzz}'