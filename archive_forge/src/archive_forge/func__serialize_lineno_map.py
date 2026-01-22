from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def _serialize_lineno_map(self, env, ccodewriter):
    tb = env.context.gdb_debug_outputwriter
    markers = ccodewriter.buffer.allmarkers()
    d = defaultdict(list)
    for c_lineno, (src_desc, src_lineno) in enumerate(markers):
        if src_lineno > 0 and src_desc.filename is not None:
            d[src_desc, src_lineno].append(c_lineno + 1)
    tb.start('LineNumberMapping')
    for (src_desc, src_lineno), c_linenos in sorted(d.items()):
        assert src_desc.filename is not None
        tb.add_entry('LineNumber', c_linenos=' '.join(map(str, c_linenos)), src_path=src_desc.filename, src_lineno=str(src_lineno))
    tb.end('LineNumberMapping')
    tb.serialize()