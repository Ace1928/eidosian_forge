import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _add_summary_step_transformer(parent, node, full_name, name, logs):
    """Adds a step argument to the summary API call if not specified.

  The inserted argument value is tf.compat.v1.train.get_or_create_global_step().
  """
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'step':
            return node
    default_value = 'tf.compat.v1.train.get_or_create_global_step()'
    ast_value = ast.parse(default_value).body[0].value
    del ast_value.lineno
    node.keywords.append(ast.keyword(arg='step', value=ast_value))
    logs.append((ast_edits.WARNING, node.lineno, node.col_offset, "Summary API writing function %s now requires a 'step' argument; inserting default of %s." % (full_name or name, default_value)))
    return node