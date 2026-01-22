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
def _fix_paths_dict(self, kw):
    for k in kw.keys():
        v = kw[k]
        if k in ['sources', 'depends', 'include_dirs', 'library_dirs', 'module_dirs', 'extra_objects']:
            new_v = self.paths(v)
            kw[k] = new_v