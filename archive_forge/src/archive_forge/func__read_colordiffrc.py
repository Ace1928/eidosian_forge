import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
def _read_colordiffrc(self, path):
    try:
        self.colors.update(read_colordiffrc(path))
    except OSError:
        pass