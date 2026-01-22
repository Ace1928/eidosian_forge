import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def _get_configuration_from_setup_py(self, setup_py, subpackage_name, subpackage_path, parent_name, caller_level=1):
    sys.path.insert(0, os.path.dirname(setup_py))
    try:
        setup_name = os.path.splitext(os.path.basename(setup_py))[0]
        n = dot_join(self.name, subpackage_name, setup_name)
        setup_module = exec_mod_from_location('_'.join(n.split('.')), setup_py)
        if not hasattr(setup_module, 'configuration'):
            if not self.options['assume_default_configuration']:
                self.warn('Assuming default configuration (%s does not define configuration())' % setup_module)
            config = Configuration(subpackage_name, parent_name, self.top_path, subpackage_path, caller_level=caller_level + 1)
        else:
            pn = dot_join(*[parent_name] + subpackage_name.split('.')[:-1])
            args = (pn,)
            if setup_module.configuration.__code__.co_argcount > 1:
                args = args + (self.top_path,)
            config = setup_module.configuration(*args)
        if config.name != dot_join(parent_name, subpackage_name):
            self.warn('Subpackage %r configuration returned as %r' % (dot_join(parent_name, subpackage_name), config.name))
    finally:
        del sys.path[0]
    return config