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
def _get_hashed_path(hashed_grammar, path, cache_path=None):
    directory = _get_cache_directory_path(cache_path=cache_path)
    file_hash = hashlib.sha256(str(path).encode('utf-8')).hexdigest()
    return os.path.join(directory, '%s-%s.pkl' % (hashed_grammar, file_hash))