from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def _find_c_source(base_path):
    file_exists = os.path.exists
    for ext in C_FILE_EXTENSIONS:
        file_name = base_path + ext
        if file_exists(file_name):
            return file_name
    return None