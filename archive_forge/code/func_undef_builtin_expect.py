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
def undef_builtin_expect(self, cond):
    """
        Redefine the macros likely() and unlikely to no-ops, depending on
        condition 'cond'
        """
    self.putln('#if %s' % cond)
    self.putln('    #undef likely')
    self.putln('    #undef unlikely')
    self.putln('    #define likely(x)   (x)')
    self.putln('    #define unlikely(x) (x)')
    self.putln('#endif')