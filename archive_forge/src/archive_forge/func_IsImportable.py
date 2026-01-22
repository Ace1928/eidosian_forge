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
def IsImportable(name, path):
    """Checks if given name can be imported at given path.

  Args:
    name: str, module name without '.' or suffixes.
    path: str, filesystem path to location of the module.

  Returns:
    True, if name is importable.
  """
    if os.path.isdir(path):
        if not os.path.isfile(os.path.join(path, '__init__.py')):
            return path in sys.path
        name_path = os.path.join(path, name)
        if os.path.isdir(name_path):
            return os.path.isfile(os.path.join(name_path, '__init__.py'))
        return os.path.exists(name_path + '.py')
    name_path = name.split('.')
    importer = pkgutil.get_importer(os.path.join(path, *name_path[:-1]))
    if not importer:
        return False
    find_spec_exists = hasattr(importer, 'find_spec')
    find_method = importer.find_spec if find_spec_exists else importer.find_module
    return find_method(name_path[-1]) is not None