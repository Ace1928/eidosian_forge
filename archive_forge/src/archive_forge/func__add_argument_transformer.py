import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _add_argument_transformer(parent, node, full_name, name, logs, arg_name, arg_value_ast):
    """Adds an argument (as a final kwarg arg_name=arg_value_ast)."""
    node.keywords.append(ast.keyword(arg=arg_name, value=arg_value_ast))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, "Adding argument '%s' to call to %s." % (pasta.dump(node.keywords[-1]), full_name or name)))
    return node