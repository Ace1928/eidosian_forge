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
def _textpage_dict(self, raw=False):
    page_dict = {'width': self.rect.width, 'height': self.rect.height}
    self._getNewBlockList(page_dict, raw)
    return page_dict