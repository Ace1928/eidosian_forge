import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
def test_unapplied_multi(self):
    self.make_empty_quilt_dir('source')
    self.build_tree_contents([('source/patches/series', 'patch1.diff\npatch2.diff'), ('source/patches/patch1.diff', 'foob ar'), ('source/patches/patch2.diff', 'bazb ar')])
    self.assertEqual(['patch1.diff', 'patch2.diff'], quilt_unapplied('source', 'patches'))