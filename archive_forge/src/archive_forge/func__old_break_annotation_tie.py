import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _old_break_annotation_tie(annotated_lines):
    """Chose an attribution between several possible ones.

    :param annotated_lines: A list of tuples ((file_id, rev_id), line) where
        the lines are identical but the revids different while no parent
        relation exist between them

     :return : The "winning" line. This must be one with a revid that
         guarantees that further criss-cross merges will converge. Failing to
         do so have performance implications.
    """
    return sorted(annotated_lines)[0]