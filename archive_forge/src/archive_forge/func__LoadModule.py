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
def _LoadModule(importer, module_path, module_name, name_to_give):
    """Loads the module or package under given name."""
    code = importer.get_code(module_name)
    module = types.ModuleType(name_to_give)
    if importer.is_package(module_name):
        module.__path__ = [module_path]
        module.__file__ = os.path.join(module_path, '__init__.pyc')
    else:
        module.__file__ = module_path + '.pyc'
    exec(code, module.__dict__)
    sys.modules[name_to_give] = module
    return module