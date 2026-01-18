from __future__ import annotations
from difflib import SequenceMatcher
from typing import Iterable, Iterator
from kombu import version_info_t
def version_string_as_tuple(s: str) -> version_info_t:
    """Convert version string to version info tuple."""
    v = _unpack_version(*s.split('.'))
    if isinstance(v.micro, str):
        v = version_info_t(v.major, v.minor, *_splitmicro(*v[2:]))
    if not v.serial and v.releaselevel and ('-' in v.releaselevel):
        v = version_info_t(*list(v[0:3]) + v.releaselevel.split('-'))
    return v