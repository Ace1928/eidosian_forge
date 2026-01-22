import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
def _glibc_version_string_confstr() -> Optional[str]:
    """
    Primary implementation of glibc_version_string using os.confstr.
    """
    try:
        version_string: str = getattr(os, 'confstr')('CS_GNU_LIBC_VERSION')
        assert version_string is not None
        _, version = version_string.rsplit()
    except (AssertionError, AttributeError, OSError, ValueError):
        return None
    return version