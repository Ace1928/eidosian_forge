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
class IdentityMatrix(Matrix):
    """Identity matrix [1, 0, 0, 1, 0, 0]"""

    def __hash__(self):
        return hash((1, 0, 0, 1, 0, 0))

    def __init__(self):
        Matrix.__init__(self, 1.0, 1.0)

    def __repr__(self):
        return 'IdentityMatrix(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)'

    def __setattr__(self, name, value):
        if name in 'ad':
            self.__dict__[name] = 1.0
        elif name in 'bcef':
            self.__dict__[name] = 0.0
        else:
            self.__dict__[name] = value

    def checkargs(*args):
        raise NotImplementedError('Identity is readonly')