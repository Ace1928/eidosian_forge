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
def NewAcl() -> Any:
    nAclLength = 32767
    acl_data = ctypes.create_string_buffer(nAclLength)
    pAcl = ctypes.cast(acl_data, PACL).contents
    advapi32.InitializeAcl(pAcl, nAclLength, ACL_REVISION)
    return pAcl