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
def LookupAccountName(lpSystemName: Any, lpAccountName: Any) -> Any:
    cbSid = wintypes.DWORD(0)
    cchReferencedDomainName = wintypes.DWORD(0)
    peUse = wintypes.DWORD(0)
    try:
        advapi32.LookupAccountNameW(lpSystemName, lpAccountName, None, ctypes.byref(cbSid), None, ctypes.byref(cchReferencedDomainName), ctypes.byref(peUse))
    except OSError as e:
        if e.winerror != ERROR_INSUFFICIENT_BUFFER:
            raise
    Sid = ctypes.create_unicode_buffer('', cbSid.value)
    pSid = ctypes.cast(ctypes.pointer(Sid), wintypes.LPVOID)
    lpReferencedDomainName = ctypes.create_unicode_buffer('', cchReferencedDomainName.value + 1)
    success = advapi32.LookupAccountNameW(lpSystemName, lpAccountName, pSid, ctypes.byref(cbSid), lpReferencedDomainName, ctypes.byref(cchReferencedDomainName), ctypes.byref(peUse))
    if not success:
        raise ctypes.WinError()
    return (pSid, lpReferencedDomainName.value, peUse.value)