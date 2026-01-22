from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import glob
import importlib.util
import os
import pkgutil
import sys
import types
from googlecloudsdk.core.util import files
def _ListPackagesAndFiles(path):
    """List packages or modules which can be imported at given path."""
    importables = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            importables.append(filename)
        else:
            pkg_init_filepath = os.path.join(path, filename, '__init__.py')
            if os.path.isfile(pkg_init_filepath):
                importables.append(os.path.join(filename, '__init__.py'))
    return importables