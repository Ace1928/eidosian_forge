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
def _wildcard_get_subpackage(self, subpackage_name, parent_name, caller_level=1):
    l = subpackage_name.split('.')
    subpackage_path = njoin([self.local_path] + l)
    dirs = [_m for _m in sorted_glob(subpackage_path) if os.path.isdir(_m)]
    config_list = []
    for d in dirs:
        if not os.path.isfile(njoin(d, '__init__.py')):
            continue
        if 'build' in d.split(os.sep):
            continue
        n = '.'.join(d.split(os.sep)[-len(l):])
        c = self.get_subpackage(n, parent_name=parent_name, caller_level=caller_level + 1)
        config_list.extend(c)
    return config_list