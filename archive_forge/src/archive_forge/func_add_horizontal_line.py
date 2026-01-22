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
def add_horizontal_line(self):
    """Add horizontal line ("hr" tag)"""
    child = self.create_element('hr')
    self.append_child(child)
    return child