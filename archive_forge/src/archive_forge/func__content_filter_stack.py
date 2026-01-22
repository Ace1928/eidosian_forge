import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def _content_filter_stack(path=None, file_id=None):
    if path.endswith('.txt'):
        return [ContentFilter(txt_reader, txt_writer)]
    else:
        return []