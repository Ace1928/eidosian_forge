import os
import collections.abc
def _parse_table_file(fd):
    for line in fd:
        line = line.rstrip()
        if not line or line.startswith('#'):
            continue
        yield line.split()