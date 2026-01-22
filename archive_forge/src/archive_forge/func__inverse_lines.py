import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def _inverse_lines(self, new_lines, file_id):
    """Produce a version with only those changes removed from new_lines."""
    target_path = self.target_tree.id2path(file_id)
    target_lines = self.target_tree.get_file_lines(target_path)
    work_path = self.work_tree.id2path(file_id)
    work_lines = self.work_tree.get_file_lines(work_path)
    import patiencediff
    from merge3 import Merge3
    return Merge3(new_lines, target_lines, work_lines, sequence_matcher=patiencediff.PatienceSequenceMatcher).merge_lines()