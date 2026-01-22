import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
def _get_module_src(path):
    if not _module_src_cache.get(path):
        with open(path, 'r') as f:
            _module_src_cache[path] = f.read()
    return _module_src_cache[path]