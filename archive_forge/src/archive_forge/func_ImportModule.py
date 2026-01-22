from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import importlib
import importlib.util
import os
import sys
from googlecloudsdk.core import exceptions
import six
def ImportModule(module_path):
    """Imports a module object given its ModulePath and returns it.

  A module_path from GetModulePath() from any valid installation is importable
  by ImportModule() in another installation of same release.

  Args:
    module_path: The module path to import.

  Raises:
    ImportModuleError: Malformed module path or any failure to import.

  Returns:
    The Cloud SDK object named by module_path.
  """
    parts = module_path.split(':')
    if len(parts) > 2:
        raise ImportModuleError('Module path [{}] must be in the form: package(.module)+(:attribute(.attribute)*)?'.format(module_path))
    try:
        module = importlib.import_module(parts[0])
    except ImportError as e:
        raise ImportModuleError('Module path [{}] not found: {}.'.format(module_path, e))
    if len(parts) == 1:
        return module
    obj = module
    attributes = parts[1].split('.')
    for attr in attributes:
        try:
            obj = getattr(obj, attr)
        except AttributeError as e:
            raise ImportModuleError('Module path [{}] not found: {}.'.format(module_path, e))
    return obj