from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
from streamlit.util import HASHLIB_KWARGS
def _dirfiles(dir_path: str, glob_pattern: str) -> str:
    p = Path(dir_path)
    filenames = sorted([f.name for f in p.glob(glob_pattern) if not f.name.startswith('.')])
    return '+'.join(filenames)