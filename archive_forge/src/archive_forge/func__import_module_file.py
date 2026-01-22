import importlib
import logging
import os
import sys
def _import_module_file(path):
    abspath = os.path.abspath(path)
    original_path = list(sys.path)
    sys.path.append(os.path.dirname(abspath))
    modname = chop_py_suffix(os.path.basename(abspath))
    try:
        return load_source(modname, abspath)
    finally:
        sys.path = original_path