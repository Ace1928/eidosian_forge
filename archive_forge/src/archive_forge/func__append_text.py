import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def _append_text(chunks, context=None):
    """A content filter that appends a string to the end of the file.

    This tests filters that change the length."""
    return chunks + [_trailer_string]