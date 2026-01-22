import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
def _is_linux_i686(executable: str) -> bool:
    with _parse_elf(executable) as f:
        return f is not None and f.capacity == EIClass.C32 and (f.encoding == EIData.Lsb) and (f.machine == EMachine.I386)