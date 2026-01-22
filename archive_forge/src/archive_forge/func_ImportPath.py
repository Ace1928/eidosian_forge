from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import importlib
import importlib.util
import os
import sys
from googlecloudsdk.core import exceptions
import six
def ImportPath(path):
    """Imports and returns the module given a python source file path."""
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec:
        raise ImportModuleError('Module file [{}] not found.'.format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportModuleError('Module file [{}] not found: {}.'.format(path, e))
    return module