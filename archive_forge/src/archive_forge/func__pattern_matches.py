import fnmatch
import getpass
import os
import re
import shlex
import socket
from hashlib import sha1
from io import StringIO
from functools import partial
from .ssh_exception import CouldNotCanonicalize, ConfigParseError
def _pattern_matches(self, patterns, target):
    if hasattr(patterns, 'split'):
        patterns = patterns.split(',')
    match = False
    for pattern in patterns:
        if pattern.startswith('!') and fnmatch.fnmatch(target, pattern[1:]):
            return False
        elif fnmatch.fnmatch(target, pattern):
            match = True
    return match