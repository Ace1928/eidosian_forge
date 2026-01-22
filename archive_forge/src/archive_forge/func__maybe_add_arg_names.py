import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _maybe_add_arg_names(self, node, full_name):
    """Make args into keyword args if function called full_name requires it."""
    function_reorders = self._api_change_spec.function_reorders
    if full_name in function_reorders:
        if uses_star_args_in_call(node):
            self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require re-ordering the call arguments, but it was passed variable-length positional *args. The upgrade script cannot handle these automatically.' % full_name)
        reordered = function_reorders[full_name]
        new_args = []
        new_keywords = []
        idx = 0
        for arg in node.args:
            if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
                continue
            keyword_arg = reordered[idx]
            if keyword_arg:
                new_keywords.append(ast.keyword(arg=keyword_arg, value=arg))
            else:
                new_args.append(arg)
            idx += 1
        if new_keywords:
            self.add_log(INFO, node.lineno, node.col_offset, 'Added keywords to args of function %r' % full_name)
            node.args = new_args
            node.keywords = new_keywords + (node.keywords or [])
            return True
    return False