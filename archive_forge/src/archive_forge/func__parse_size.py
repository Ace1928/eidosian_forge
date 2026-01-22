import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def _parse_size(size_str):
    suffixes = {'': 1000000.0, 'b': 1.0, 'k': 1000.0, 'M': 1000000.0, 'G': 1000000000.0, 'T': 1000000000000.0, 'kb': 1000.0, 'Mb': 1000000.0, 'Gb': 1000000000.0, 'Tb': 1000000000000.0, 'kib': 1024.0, 'Mib': 1024.0 ** 2, 'Gib': 1024.0 ** 3, 'Tib': 1024.0 ** 4}
    m = re.match('^\\s*(\\d+)\\s*({})\\s*$'.format('|'.join(suffixes.keys())), size_str, re.I)
    if not m or m.group(2) not in suffixes:
        raise ValueError('Invalid size string')
    return float(m.group(1)) * suffixes[m.group(2)]