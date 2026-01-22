import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _name_scope_transformer(parent, node, full_name, name, logs):
    """Fix name scope invocation to use 'default_name' and omit 'values' args."""
    name_found, name = ast_edits.get_arg_value(node, 'name', 0)
    default_found, default_name = ast_edits.get_arg_value(node, 'default_name', 1)
    if name_found and pasta.dump(name) != 'None':
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, '`name` passed to `name_scope`. Because you may be re-entering an existing scope, it is not safe to convert automatically,  the v2 name_scope does not support re-entering scopes by name.\n'))
        new_name = 'tf.compat.v1.name_scope'
        logs.append((ast_edits.INFO, node.func.lineno, node.func.col_offset, 'Renamed %r to %r' % (full_name, new_name)))
        new_name_node = ast_edits.full_name_node(new_name, node.func.ctx)
        ast.copy_location(new_name_node, node.func)
        pasta.ast_utils.replace_child(node, node.func, new_name_node)
        return node
    if default_found:
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Using default_name as name in call to name_scope.\n'))
        node.args = []
        node.keywords = [ast.keyword(arg='name', value=default_name)]
        return node
    logs.append((ast_edits.ERROR, node.lineno, node.col_offset, 'name_scope call with neither name nor default_name cannot be converted properly.'))