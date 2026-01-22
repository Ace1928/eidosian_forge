import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _rename_func(node, full_name, new_name, logs, reason):
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Renamed %r to %r: %s' % (full_name, new_name, reason)))
    new_name_node = ast_edits.full_name_node(new_name, node.func.ctx)
    ast.copy_location(new_name_node, node.func)
    pasta.ast_utils.replace_child(node, node.func, new_name_node)
    return node