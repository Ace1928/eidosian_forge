from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def GetFileSecurity(lpFileName: Any, RequestedInformation: Any) -> Any:
    nLength = wintypes.DWORD(0)
    try:
        advapi32.GetFileSecurityW(lpFileName, RequestedInformation, None, 0, ctypes.byref(nLength))
    except OSError as e:
        if e.winerror != ERROR_INSUFFICIENT_BUFFER:
            raise
    if not nLength.value:
        return None
    pSecurityDescriptor = (wintypes.BYTE * nLength.value)()
    advapi32.GetFileSecurityW(lpFileName, RequestedInformation, pSecurityDescriptor, nLength, ctypes.byref(nLength))
    return pSecurityDescriptor