from test import support
from test.support import import_helper
import_helper.import_module('_multiprocessing')
import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile
from test.support import os_helper
from test.support.script_helper import (
import multiprocess as multiprocessing
import_helper.import_module('multiprocess.synchronize')
import sys
import time
from multiprocess import Pool, set_start_method
import sys
import time
from multiprocess import Pool, set_start_method
import sys, os.path, runpy
def _make_test_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename, source=test_source, depth=1):
    to_return = make_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename, source, depth)
    importlib.invalidate_caches()
    return to_return