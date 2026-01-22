import re
import sys
from os.path import expanduser
import patiencediff
from . import terminal, trace
from .commands import get_cmd_object
from .patches import (ContextLine, Hunk, HunkLine, InsertLine, RemoveLine,
@staticmethod
def _matched_lines(old, new):
    matcher = patiencediff.PatienceSequenceMatcher(None, old, new)
    matched_lines = sum((n for i, j, n in matcher.get_matching_blocks()))
    return matched_lines