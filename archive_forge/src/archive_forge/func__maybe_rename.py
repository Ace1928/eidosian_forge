import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _maybe_rename(self, parent, node, full_name):
    """Replace node (Attribute or Name) with a node representing full_name."""
    new_name = self._api_change_spec.symbol_renames.get(full_name, None)
    if new_name:
        self.add_log(INFO, node.lineno, node.col_offset, 'Renamed %r to %r' % (full_name, new_name))
        new_node = full_name_node(new_name, node.ctx)
        ast.copy_location(new_node, node)
        pasta.ast_utils.replace_child(parent, node, new_node)
        return True
    else:
        return False