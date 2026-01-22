imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
class PyxArgs(object):
    build_dir = True
    build_in_temp = True
    setup_args = {}