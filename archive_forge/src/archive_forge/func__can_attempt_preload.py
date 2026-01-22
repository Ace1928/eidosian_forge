import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _can_attempt_preload(lib: str) -> bool:
    """Returns if the preload can be attempted."""
    config = get_preload_config()
    if config is None or config['packaging'] == 'conda':
        _log(f'Cannot preload {lib} as this is not a wheel installation')
        return False
    if lib not in _preload_libs:
        raise AssertionError(f'Unknown preload library: {lib}')
    if lib not in config:
        _log(f'Preload {lib} not configured in wheel')
        return False
    if _preload_libs[lib] is not None:
        _log(f'Preload already attempted: {lib}')
        return False
    return True