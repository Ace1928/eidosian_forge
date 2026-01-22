from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def _find_c_source_files(self, dir_path, source_file):
    """
        Desperately parse all C files in the directory or its package parents
        (not re-descending) to find the (included) source file in one of them.
        """
    if not os.path.isdir(dir_path):
        return
    splitext = os.path.splitext
    for filename in os.listdir(dir_path):
        ext = splitext(filename)[1].lower()
        if ext in C_FILE_EXTENSIONS:
            self._read_source_lines(os.path.join(dir_path, filename), source_file)
            if source_file in self._c_files_map:
                return
    if is_package_dir(dir_path):
        self._find_c_source_files(os.path.dirname(dir_path), source_file)