import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
def test_unapplied_dir(self):
    self.make_empty_quilt_dir('source')
    self.build_tree_contents([('source/patches/series', 'debian/patch1.diff\n'), ('source/patches/debian/',), ('source/patches/debian/patch1.diff', 'foob ar')])
    self.assertEqual(['debian/patch1.diff'], quilt_unapplied('source'))