import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def _mk_file(self, path, line_prefix, total_lines, versioned):
    text = ''
    for i in range(total_lines):
        text += line_prefix + str(i + 1) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    if versioned:
        self.run_bzr(['add', path])
        self.run_bzr(['ci', '-m', '"' + path + '"'])