import email
import itertools
import functools
import os
import posixpath
import re
import zipfile
import contextlib
from distutils.util import get_platform
import setuptools
from setuptools.extern.packaging.version import Version as parse_version
from setuptools.extern.packaging.tags import sys_tags
from setuptools.extern.packaging.utils import canonicalize_name
from setuptools.command.egg_info import write_requirements, _egg_basename
from setuptools.archive_util import _unpack_zipfile_obj
@staticmethod
def _move_data_entries(destination_eggdir, dist_data):
    """Move data entries to their correct location."""
    dist_data = os.path.join(destination_eggdir, dist_data)
    dist_data_scripts = os.path.join(dist_data, 'scripts')
    if os.path.exists(dist_data_scripts):
        egg_info_scripts = os.path.join(destination_eggdir, 'EGG-INFO', 'scripts')
        os.mkdir(egg_info_scripts)
        for entry in os.listdir(dist_data_scripts):
            if entry.endswith('.pyc'):
                os.unlink(os.path.join(dist_data_scripts, entry))
            else:
                os.rename(os.path.join(dist_data_scripts, entry), os.path.join(egg_info_scripts, entry))
        os.rmdir(dist_data_scripts)
    for subdir in filter(os.path.exists, (os.path.join(dist_data, d) for d in ('data', 'headers', 'purelib', 'platlib'))):
        unpack(subdir, destination_eggdir)
    if os.path.exists(dist_data):
        os.rmdir(dist_data)