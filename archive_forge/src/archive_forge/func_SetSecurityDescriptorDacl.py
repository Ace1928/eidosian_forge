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
def SetSecurityDescriptorDacl(pSecurityDescriptor: Any, bDaclPresent: Any, pDacl: Any, bDaclDefaulted: Any) -> Any:
    advapi32.SetSecurityDescriptorDacl(pSecurityDescriptor, bDaclPresent, pDacl, bDaclDefaulted)