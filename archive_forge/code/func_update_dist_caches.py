from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
def update_dist_caches(dist_path, fix_zipimporter_caches):
    """
    Fix any globally cached `dist_path` related data

    `dist_path` should be a path of a newly installed egg distribution (zipped
    or unzipped).

    sys.path_importer_cache contains finder objects that have been cached when
    importing data from the original distribution. Any such finders need to be
    cleared since the replacement distribution might be packaged differently,
    e.g. a zipped egg distribution might get replaced with an unzipped egg
    folder or vice versa. Having the old finders cached may then cause Python
    to attempt loading modules from the replacement distribution using an
    incorrect loader.

    zipimport.zipimporter objects are Python loaders charged with importing
    data packaged inside zip archives. If stale loaders referencing the
    original distribution, are left behind, they can fail to load modules from
    the replacement distribution. E.g. if an old zipimport.zipimporter instance
    is used to load data from a new zipped egg archive, it may cause the
    operation to attempt to locate the requested data in the wrong location -
    one indicated by the original distribution's zip archive directory
    information. Such an operation may then fail outright, e.g. report having
    read a 'bad local file header', or even worse, it may fail silently &
    return invalid data.

    zipimport._zip_directory_cache contains cached zip archive directory
    information for all existing zipimport.zipimporter instances and all such
    instances connected to the same archive share the same cached directory
    information.

    If asked, and the underlying Python implementation allows it, we can fix
    all existing zipimport.zipimporter instances instead of having to track
    them down and remove them one by one, by updating their shared cached zip
    archive directory information. This, of course, assumes that the
    replacement distribution is packaged as a zipped egg.

    If not asked to fix existing zipimport.zipimporter instances, we still do
    our best to clear any remaining zipimport.zipimporter related cached data
    that might somehow later get used when attempting to load data from the new
    distribution and thus cause such load operations to fail. Note that when
    tracking down such remaining stale data, we can not catch every conceivable
    usage from here, and we clear only those that we know of and have found to
    cause problems if left alive. Any remaining caches should be updated by
    whomever is in charge of maintaining them, i.e. they should be ready to
    handle us replacing their zip archives with new distributions at runtime.

    """
    normalized_path = normalize_path(dist_path)
    _uncache(normalized_path, sys.path_importer_cache)
    if fix_zipimporter_caches:
        _replace_zip_directory_cache_data(normalized_path)
    else:
        _remove_and_clear_zip_directory_cache_data(normalized_path)