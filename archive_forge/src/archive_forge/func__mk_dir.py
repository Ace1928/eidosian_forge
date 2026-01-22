import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def _mk_dir(self, path, versioned):
    os.mkdir(path)
    if versioned:
        self.run_bzr(['add', path])
        self.run_bzr(['ci', '-m', '"' + path + '"'])