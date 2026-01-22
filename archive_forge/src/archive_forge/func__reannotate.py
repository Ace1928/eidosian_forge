import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _reannotate(parent_lines, new_lines, new_revision_id, matching_blocks=None):
    new_cur = 0
    if matching_blocks is None:
        plain_parent_lines = [l for r, l in parent_lines]
        matcher = patiencediff.PatienceSequenceMatcher(None, plain_parent_lines, new_lines)
        matching_blocks = matcher.get_matching_blocks()
    lines = []
    for i, j, n in matching_blocks:
        for line in new_lines[new_cur:j]:
            lines.append((new_revision_id, line))
        lines.extend(parent_lines[i:i + n])
        new_cur = j + n
    return lines