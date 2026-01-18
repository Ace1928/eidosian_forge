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
def uri_to_dict(uri):
    items = self.uri[1:].split('&')
    ret = dict()
    for item in items:
        eq = item.find('=')
        if eq >= 0:
            ret[item[:eq]] = item[eq + 1:]
        else:
            ret[item] = None
    return ret