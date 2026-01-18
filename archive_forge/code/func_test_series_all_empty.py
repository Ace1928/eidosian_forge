import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
def test_series_all_empty(self):
    source = self.make_empty_quilt_dir('source')
    self.assertEqual([], quilt_series(source, 'patches/series'))