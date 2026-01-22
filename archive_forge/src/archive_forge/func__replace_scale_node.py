import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _replace_scale_node(parent, old_value):
    """Replaces old_value with 0.5*(old_value)."""
    half = ast.Num(n=0.5)
    half.lineno = 0
    half.col_offset = 0
    new_value = ast.BinOp(left=half, op=ast.Mult(), right=old_value)
    pasta.ast_utils.replace_child(parent, old_value, new_value)
    pasta.base.formatting.set(old_value, 'prefix', '(')
    pasta.base.formatting.set(old_value, 'suffix', ')')