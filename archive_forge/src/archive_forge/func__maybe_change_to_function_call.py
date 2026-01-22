import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _maybe_change_to_function_call(self, parent, node, full_name):
    """Wraps node (typically, an Attribute or Expr) in a Call."""
    if full_name in self._api_change_spec.change_to_function:
        if not isinstance(parent, ast.Call):
            new_node = ast.Call(node, [], [])
            pasta.ast_utils.replace_child(parent, node, new_node)
            ast.copy_location(new_node, node)
            self.add_log(INFO, node.lineno, node.col_offset, 'Changed %r to a function call' % full_name)
            return True
    return False