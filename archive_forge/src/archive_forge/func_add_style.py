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
def add_style(self, text):
    """Set some style via CSS style. Replaces complete style spec."""
    style = self.get_attribute_value('style')
    if style is not None and text in style:
        return self
    self.remove_attribute('style')
    if style is None:
        style = text
    else:
        style += ';' + text
    self.set_attribute('style', style)
    return self