import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _maybe_add_module_deprecation_warning(self, node, full_name, whole_name):
    """Adds a warning if full_name is a deprecated module."""
    warnings = self._api_change_spec.module_deprecations
    if full_name in warnings:
        level, message = warnings[full_name]
        message = message.replace('<function name>', whole_name)
        self.add_log(level, node.lineno, node.col_offset, 'Using member %s in deprecated module %s. %s' % (whole_name, full_name, message))
        return True
    else:
        return False