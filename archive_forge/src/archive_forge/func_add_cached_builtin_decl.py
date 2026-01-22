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
def add_cached_builtin_decl(self, entry):
    if entry.is_builtin and entry.is_const:
        if self.should_declare(entry.cname, entry):
            self.put_pyobject_decl(entry)
            w = self.parts['cached_builtins']
            condition = None
            if entry.name in non_portable_builtins_map:
                condition, replacement = non_portable_builtins_map[entry.name]
                w.putln('#if %s' % condition)
                self.put_cached_builtin_init(entry.pos, StringEncoding.EncodedString(replacement), entry.cname)
                w.putln('#else')
            self.put_cached_builtin_init(entry.pos, StringEncoding.EncodedString(entry.name), entry.cname)
            if condition:
                w.putln('#endif')