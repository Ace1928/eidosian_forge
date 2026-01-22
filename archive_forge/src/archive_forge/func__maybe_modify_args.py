import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _maybe_modify_args(self, node, full_name, name):
    """Rename keyword args if the function called full_name requires it."""
    renamed_keywords = self._get_applicable_dict('function_keyword_renames', full_name, name)
    if not renamed_keywords:
        return False
    if uses_star_kwargs_in_call(node):
        self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require renaming or removing call arguments, but it was passed variable-length *args or **kwargs. The upgrade script cannot handle these automatically.' % (full_name or name))
    modified = False
    new_keywords = []
    for keyword in node.keywords:
        argkey = keyword.arg
        if argkey in renamed_keywords:
            modified = True
            if renamed_keywords[argkey] is None:
                lineno = getattr(keyword, 'lineno', node.lineno)
                col_offset = getattr(keyword, 'col_offset', node.col_offset)
                self.add_log(INFO, lineno, col_offset, 'Removed argument %s for function %s' % (argkey, full_name or name))
            else:
                keyword.arg = renamed_keywords[argkey]
                lineno = getattr(keyword, 'lineno', node.lineno)
                col_offset = getattr(keyword, 'col_offset', node.col_offset)
                self.add_log(INFO, lineno, col_offset, 'Renamed keyword argument for %s from %s to %s' % (full_name, argkey, renamed_keywords[argkey]))
                new_keywords.append(keyword)
        else:
            new_keywords.append(keyword)
    if modified:
        node.keywords = new_keywords
    return modified