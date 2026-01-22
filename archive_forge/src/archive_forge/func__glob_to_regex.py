from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
def _glob_to_regex(pattern: str) -> str:
    """Convert a file-path glob pattern into a regex."""
    pattern = pattern.replace('\\', '/')
    if '/' not in pattern:
        pattern = '**/' + pattern
    path_rx = []
    pos = 0
    while pos < len(pattern):
        for rx, sub in G2RX_TOKENS:
            if (m := rx.match(pattern, pos=pos)):
                if sub is None:
                    raise ConfigError(f"File pattern can't include {m[0]!r}")
                path_rx.append(m.expand(sub))
                pos = m.end()
                break
    return ''.join(path_rx)