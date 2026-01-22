import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def _fix_paths(paths, local_path, include_non_existing):
    assert is_sequence(paths), repr(type(paths))
    new_paths = []
    assert not is_string(paths), repr(paths)
    for n in paths:
        if is_string(n):
            if '*' in n or '?' in n:
                p = sorted_glob(n)
                p2 = sorted_glob(njoin(local_path, n))
                if p2:
                    new_paths.extend(p2)
                elif p:
                    new_paths.extend(p)
                else:
                    if include_non_existing:
                        new_paths.append(n)
                    print('could not resolve pattern in %r: %r' % (local_path, n))
            else:
                n2 = njoin(local_path, n)
                if os.path.exists(n2):
                    new_paths.append(n2)
                else:
                    if os.path.exists(n):
                        new_paths.append(n)
                    elif include_non_existing:
                        new_paths.append(n)
                    if not os.path.exists(n):
                        print('non-existing path in %r: %r' % (local_path, n))
        elif is_sequence(n):
            new_paths.extend(_fix_paths(n, local_path, include_non_existing))
        else:
            new_paths.append(n)
    return [minrelpath(p) for p in new_paths]