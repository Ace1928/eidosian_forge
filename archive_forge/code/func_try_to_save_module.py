import time
import os
import sys
import hashlib
import gc
import shutil
import platform
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, Any
def try_to_save_module(hashed_grammar, file_io, module, lines, pickling=True, cache_path=None):
    path = file_io.path
    try:
        p_time = None if path is None else file_io.get_last_modified()
    except OSError:
        p_time = None
        pickling = False
    item = _NodeCacheItem(module, lines, p_time)
    _set_cache_item(hashed_grammar, path, item)
    if pickling and path is not None:
        try:
            _save_to_file_system(hashed_grammar, path, item, cache_path=cache_path)
        except PermissionError:
            warnings.warn('Tried to save a file to %s, but got permission denied.' % path, Warning)
        else:
            _remove_cache_and_update_lock(cache_path=cache_path)