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
class ClosureTempAllocator(object):

    def __init__(self, klass):
        self.klass = klass
        self.temps_allocated = {}
        self.temps_free = {}
        self.temps_count = 0

    def reset(self):
        for type, cnames in self.temps_allocated.items():
            self.temps_free[type] = list(cnames)

    def allocate_temp(self, type):
        if type not in self.temps_allocated:
            self.temps_allocated[type] = []
            self.temps_free[type] = []
        elif self.temps_free[type]:
            return self.temps_free[type].pop(0)
        cname = '%s%d' % (Naming.codewriter_temp_prefix, self.temps_count)
        self.klass.declare_var(pos=None, name=cname, cname=cname, type=type, is_cdef=True)
        self.temps_allocated[type].append(cname)
        self.temps_count += 1
        return cname