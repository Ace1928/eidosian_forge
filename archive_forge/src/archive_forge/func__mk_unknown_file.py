import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def _mk_unknown_file(self, path, line_prefix='line', total_lines=10):
    self._mk_file(path, line_prefix, total_lines, versioned=False)