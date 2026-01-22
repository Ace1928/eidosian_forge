import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def _patch_distribution_metadata():
    from . import _core_metadata
    'Patch write_pkg_file and read_pkg_file for higher metadata standards'
    for attr in ('write_pkg_info', 'write_pkg_file', 'read_pkg_file', 'get_metadata_version'):
        new_val = getattr(_core_metadata, attr)
        setattr(distutils.dist.DistributionMetadata, attr, new_val)