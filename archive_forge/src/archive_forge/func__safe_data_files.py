from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib
from itertools import chain
from .._importlib import metadata
from .build import _ORIGINAL_SUBCOMMANDS
def _safe_data_files(self, build_py):
    """
        Since the ``sdist`` class is also used to compute the MANIFEST
        (via :obj:`setuptools.command.egg_info.manifest_maker`),
        there might be recursion problems when trying to obtain the list of
        data_files and ``include_package_data=True`` (which in turn depends on
        the files included in the MANIFEST).

        To avoid that, ``manifest_maker`` should be able to overwrite this
        method and avoid recursive attempts to build/analyze the MANIFEST.
        """
    return build_py.data_files