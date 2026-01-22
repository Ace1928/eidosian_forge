import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _get_package_init(package_path):
    """Get path of __init__ file from package_path or None if not a package."""
    init_path = osutils.pathjoin(package_path, '__init__.py')
    if os.path.exists(init_path):
        return init_path
    init_path = init_path[:-3] + COMPILED_EXT
    if os.path.exists(init_path):
        return init_path
    return None