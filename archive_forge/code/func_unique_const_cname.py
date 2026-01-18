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
def unique_const_cname(self, format_str):
    used = self.const_cnames_used
    cname = value = format_str.format(sep='', counter='')
    while cname in used:
        counter = used[value] = used[value] + 1
        cname = format_str.format(sep='_', counter=counter)
    used[cname] = 1
    return cname