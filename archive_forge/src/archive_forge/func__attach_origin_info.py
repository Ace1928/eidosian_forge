import collections
import difflib
import io
import os
import tokenize
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.util import tf_inspect
def _attach_origin_info(self, node):
    lineno = getattr(node, 'lineno', None)
    col_offset = getattr(node, 'col_offset', None)
    if lineno is None:
        return
    if self._function_stack:
        function_name = self._function_stack[-1].name
    else:
        function_name = None
    source_code_line = self._source_lines[lineno - 1]
    comment = self._comments_map.get(lineno)
    loc = Location(self._filepath, self._absolute_lineno(lineno), self._absolute_col_offset(col_offset))
    origin = OriginInfo(loc, function_name, source_code_line, comment)
    anno.setanno(node, 'lineno', lineno)
    anno.setanno(node, anno.Basic.ORIGIN, origin)