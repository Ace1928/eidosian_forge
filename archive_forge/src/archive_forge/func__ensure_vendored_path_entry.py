from __future__ import (absolute_import, division, print_function)
import os
import pkgutil
import sys
import warnings
def _ensure_vendored_path_entry():
    """
    Ensure that any downstream-bundled content beneath this package is available at the top of sys.path
    """
    vendored_path_entry = os.path.dirname(__file__)
    vendored_module_names = set((m[1] for m in pkgutil.iter_modules([vendored_path_entry], '')))
    if vendored_module_names:
        if vendored_path_entry in sys.path:
            sys.path.remove(vendored_path_entry)
        sys.path.insert(0, vendored_path_entry)
        already_loaded_vendored_modules = set(sys.modules.keys()).intersection(vendored_module_names)
        if already_loaded_vendored_modules:
            warnings.warn('One or more Python packages bundled by this ansible-core distribution were already loaded ({0}). This may result in undefined behavior.'.format(', '.join(sorted(already_loaded_vendored_modules))))