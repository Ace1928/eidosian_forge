import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def text_merge(self, trans_id, paths):
    self.tt.create_file([b'qux'], trans_id)