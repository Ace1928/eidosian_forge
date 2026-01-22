from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _posix_split_name(self, name, encoding, errors):
    """Split a name longer than 100 chars into a prefix
           and a name part.
        """
    components = name.split('/')
    for i in range(1, len(components)):
        prefix = '/'.join(components[:i])
        name = '/'.join(components[i:])
        if len(prefix.encode(encoding, errors)) <= LENGTH_PREFIX and len(name.encode(encoding, errors)) <= LENGTH_NAME:
            break
    else:
        raise ValueError('name is too long')
    return (prefix, name)