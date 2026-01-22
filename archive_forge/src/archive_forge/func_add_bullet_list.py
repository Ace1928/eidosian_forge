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
def add_bullet_list(self):
    """Add bulleted list ("ul" tag)"""
    child = self.create_element('ul')
    self.append_child(child)
    return child