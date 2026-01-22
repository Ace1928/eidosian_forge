from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class PyStringConst(object):
    """Global info about a Python string constant held by GlobalState.
    """

    def __init__(self, cname, encoding, is_unicode, is_str=False, py3str_cstring=None, intern=False):
        self.cname = cname
        self.py3str_cstring = py3str_cstring
        self.encoding = encoding
        self.is_str = is_str
        self.is_unicode = is_unicode
        self.intern = intern

    def __lt__(self, other):
        return self.cname < other.cname