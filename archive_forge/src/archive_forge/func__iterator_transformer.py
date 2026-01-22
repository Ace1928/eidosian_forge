import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _iterator_transformer(parent, node, full_name, name, logs):
    """Transform iterator methods to compat function calls."""
    if full_name and (full_name.startswith('tf.compat.v1.data') or full_name.startswith('tf.data')):
        return
    if not isinstance(node.func, ast.Attribute):
        return
    node.args = [node.func.value] + node.args
    node.func.value = ast_edits.full_name_node('tf.compat.v1.data')
    logs.append((ast_edits.WARNING, node.lineno, node.col_offset, 'Changing dataset.%s() to tf.compat.v1.data.%s(dataset). Please check this transformation.\n' % (name, name)))
    return node