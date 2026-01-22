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
def JM_TUPLE3(o: typing.Sequence) -> tuple:
    return tuple(map(lambda x: round(x, 3) if abs(x) >= 0.001 else 0, o))