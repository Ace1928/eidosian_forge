import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
def _have_compatible_abi(executable: str, arch: str) -> bool:
    if arch == 'armv7l':
        return _is_linux_armhf(executable)
    if arch == 'i686':
        return _is_linux_i686(executable)
    return arch in {'x86_64', 'aarch64', 'ppc64', 'ppc64le', 's390x'}