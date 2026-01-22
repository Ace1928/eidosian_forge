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
def _add_library(self, name, sources, install_dir, build_info):
    """Common implementation for add_library and add_installed_library. Do
        not use directly"""
    build_info = copy.copy(build_info)
    build_info['sources'] = sources
    if not 'depends' in build_info:
        build_info['depends'] = []
    self._fix_paths_dict(build_info)
    self.libraries.append((name, build_info))