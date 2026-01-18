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
def zap_pyfiles(self):
    log.info('Removing .py files from temporary directory')
    for base, dirs, files in walk_egg(self.bdist_dir):
        for name in files:
            path = os.path.join(base, name)
            if name.endswith('.py'):
                log.debug('Deleting %s', path)
                os.unlink(path)
            if base.endswith('__pycache__'):
                path_old = path
                pattern = '(?P<name>.+)\\.(?P<magic>[^.]+)\\.pyc'
                m = re.match(pattern, name)
                path_new = os.path.join(base, os.pardir, m.group('name') + '.pyc')
                log.info('Renaming file from [%s] to [%s]' % (path_old, path_new))
                try:
                    os.remove(path_new)
                except OSError:
                    pass
                os.rename(path_old, path_new)