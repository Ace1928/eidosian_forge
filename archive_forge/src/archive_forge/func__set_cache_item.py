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
def _set_cache_item(hashed_grammar, path, module_cache_item):
    if sum((len(v) for v in parser_cache.values())) >= _CACHED_SIZE_TRIGGER:
        cutoff_time = time.time() - _CACHED_FILE_MINIMUM_SURVIVAL
        for key, path_to_item_map in parser_cache.items():
            parser_cache[key] = {path: node_item for path, node_item in path_to_item_map.items() if node_item.last_used > cutoff_time}
    parser_cache.setdefault(hashed_grammar, {})[path] = module_cache_item