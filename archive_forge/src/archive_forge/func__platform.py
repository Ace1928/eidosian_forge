import collections
import os
import re
import sys
import functools
import itertools
def _platform(*args):
    """ Helper to format the platform string in a filename
        compatible format e.g. "system-version-machine".
    """
    platform = '-'.join((x.strip() for x in filter(len, args)))
    platform = platform.replace(' ', '_')
    platform = platform.replace('/', '-')
    platform = platform.replace('\\', '-')
    platform = platform.replace(':', '-')
    platform = platform.replace(';', '-')
    platform = platform.replace('"', '-')
    platform = platform.replace('(', '-')
    platform = platform.replace(')', '-')
    platform = platform.replace('unknown', '')
    while 1:
        cleaned = platform.replace('--', '-')
        if cleaned == platform:
            break
        platform = cleaned
    while platform[-1] == '-':
        platform = platform[:-1]
    return platform