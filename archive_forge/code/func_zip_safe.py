from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
def zip_safe(self):
    safe = getattr(self.distribution, 'zip_safe', None)
    if safe is not None:
        return safe
    log.warn('zip_safe flag not set; analyzing archive contents...')
    return analyze_egg(self.bdist_dir, self.stubs)