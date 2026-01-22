from __future__ import absolute_import
import re
import os.path
import sys
from collections import defaultdict
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter  # requires coverage.py 4.0+
from coverage.files import canonical_filename
from .Utils import find_root_package_dir, is_package_dir, is_cython_generated_file, open_source_file
from . import __version__
def _iter_source_tokens(self):
    current_line = 1
    for line_no, code_line in sorted(self._code.items()):
        while line_no > current_line:
            yield []
            current_line += 1
        yield [('txt', code_line)]
        current_line += 1