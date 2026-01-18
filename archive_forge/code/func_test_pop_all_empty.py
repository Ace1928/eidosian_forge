import os
from ....tests import TestCaseWithTransport
from ..wrapper import (quilt_applied, quilt_delete, quilt_pop_all,
from . import quilt_feature
def test_pop_all_empty(self):
    self.make_empty_quilt_dir('source')
    quilt_pop_all('source', quiet=True)