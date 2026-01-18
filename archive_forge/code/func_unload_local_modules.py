import fnmatch
import logging
import os
import sys
import types
from typing import Optional, Set
def unload_local_modules(target_dir_path: str='.'):
    """ Unload all modules that are in the target directory or in a subdirectory of it.
    It is necessary to unload modules before re-executing a script that imports the modules,
    so that the new version of the modules is loaded.
    The module unloading feature is extracted from Streamlit's LocalSourcesWatcher (https://github.com/streamlit/streamlit/blob/1.24.0/lib/streamlit/watcher/local_sources_watcher.py)
    and packaged as a standalone function.
    """
    target_dir_path = os.path.abspath(target_dir_path)
    loaded_modules = {}
    module_paths = {name: get_module_paths(module) for name, module in dict(sys.modules).items()}
    for name, paths in module_paths.items():
        for path in paths:
            if file_is_in_folder_glob(path, target_dir_path) or file_in_pythonpath(path):
                loaded_modules[path] = name
    for module_name in loaded_modules.values():
        if module_name is not None and module_name in sys.modules:
            del sys.modules[module_name]