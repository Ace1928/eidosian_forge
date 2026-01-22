import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _get_distribution(old_value):
    """Returns an AST matching the following:
    ("uniform" if (old_value) else "truncated_normal")
    """
    dist = pasta.parse('"uniform" if old_value else "truncated_normal"')
    ifexpr = dist.body[0].value
    pasta.ast_utils.replace_child(ifexpr, ifexpr.test, old_value)
    pasta.base.formatting.set(dist, 'prefix', '(')
    pasta.base.formatting.set(dist, 'suffix', ')')
    return dist