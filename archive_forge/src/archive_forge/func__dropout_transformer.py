import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _dropout_transformer(parent, node, full_name, name, logs):
    """Replace keep_prob with 1-rate."""

    def _replace_keep_prob_node(parent, old_value):
        """Replaces old_value with 1-(old_value)."""
        one = ast.Num(n=1)
        one.lineno = 0
        one.col_offset = 0
        new_value = ast.BinOp(left=one, op=ast.Sub(), right=old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        ast.copy_location(new_value, old_value)
        pasta.base.formatting.set(old_value, 'prefix', '(')
        pasta.base.formatting.set(old_value, 'suffix', ')')
    for keep_prob in node.keywords:
        if keep_prob.arg == 'keep_prob':
            logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing keep_prob arg of tf.nn.dropout to rate\n'))
            keep_prob.arg = 'rate'
            _replace_keep_prob_node(keep_prob, keep_prob.value)
            return node
    if len(node.args) < 2:
        logs.append((ast_edits.ERROR, node.lineno, node.col_offset, 'tf.nn.dropout called without arguments, so automatic fix was disabled. tf.nn.dropout has changed the semantics of the second argument.'))
    else:
        rate_arg = ast.keyword(arg='rate', value=node.args[1])
        _replace_keep_prob_node(rate_arg, rate_arg.value)
        node.keywords.append(rate_arg)
        del node.args[1]
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.\n'))
        return node