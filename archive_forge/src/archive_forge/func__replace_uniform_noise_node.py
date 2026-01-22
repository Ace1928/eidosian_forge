import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _replace_uniform_noise_node(parent, old_value):
    """Replaces old_value with 'uniform' or 'gaussian'."""
    uniform = ast.Str(s='uniform')
    gaussian = ast.Str(s='gaussian')
    new_value = ast.IfExp(body=uniform, test=old_value, orelse=gaussian)
    pasta.ast_utils.replace_child(parent, old_value, new_value)
    ast.copy_location(new_value, old_value)
    pasta.base.formatting.set(new_value.test, 'prefix', '(')
    pasta.base.formatting.set(new_value.test, 'suffix', ')')