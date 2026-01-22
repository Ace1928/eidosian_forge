import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Optional
from ._elffile import ELFFile
def _parse_musl_version(output: str) -> Optional[_MuslVersion]:
    lines = [n for n in (n.strip() for n in output.splitlines()) if n]
    if len(lines) < 2 or lines[0][:4] != 'musl':
        return None
    m = re.match('Version (\\d+)\\.(\\d+)', lines[1])
    if not m:
        return None
    return _MuslVersion(major=int(m.group(1)), minor=int(m.group(2)))