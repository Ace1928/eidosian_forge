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
def CheckQuad(q: typing.Any) -> bool:
    """Check whether an object is convex, not empty  quad-like.

    It must be a sequence of 4 number pairs.
    """
    try:
        q0 = Quad(q)
    except Exception:
        if g_exceptions_verbose > 1:
            exception_info()
        return False
    return q0.is_convex