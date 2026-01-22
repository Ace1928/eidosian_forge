import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
def _check_call_names(self, call_node, names):
    if isinstance(call_node, ast.Call):
        if isinstance(call_node.func, ast.Name):
            if call_node.func.id in names:
                return True
    return False