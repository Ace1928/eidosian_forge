import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _reannotate_annotated(right_parent_lines, new_lines, new_revision_id, annotated_lines, heads_provider):
    """Update the annotations for a node based on another parent.

    :param right_parent_lines: A list of annotated lines for the right-hand
        parent.
    :param new_lines: The unannotated new lines.
    :param new_revision_id: The revision_id to attribute to lines which are not
        present in either parent.
    :param annotated_lines: A list of annotated lines. This should be the
        annotation of new_lines based on parents seen so far.
    :param heads_provider: When parents disagree on the lineage of a line, we
        need to check if one side supersedes the other.
    """
    if len(new_lines) != len(annotated_lines):
        raise AssertionError('mismatched new_lines and annotated_lines')
    lines = []
    lines_extend = lines.extend
    last_right_idx = 0
    last_left_idx = 0
    matching_left_and_right = _get_matching_blocks(right_parent_lines, annotated_lines)
    for right_idx, left_idx, match_len in matching_left_and_right:
        if last_right_idx == right_idx or last_left_idx == left_idx:
            lines_extend(annotated_lines[last_left_idx:left_idx])
        else:
            _find_matching_unannotated_lines(lines, new_lines, annotated_lines, last_left_idx, left_idx, right_parent_lines, last_right_idx, right_idx, heads_provider, new_revision_id)
        last_right_idx = right_idx + match_len
        last_left_idx = left_idx + match_len
        lines_extend(annotated_lines[left_idx:left_idx + match_len])
    return lines