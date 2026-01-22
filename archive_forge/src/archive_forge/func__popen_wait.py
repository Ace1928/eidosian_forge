from __future__ import annotations
import os
import subprocess
import sys
import warnings
from typing import Any, Optional, Sequence
def _popen_wait(popen: subprocess.Popen[Any], timeout: Optional[float]) -> Optional[int]:
    """Implement wait timeout support for Python 3."""
    try:
        return popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None