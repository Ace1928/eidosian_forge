import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_recursive_diff(self):
    """Children of directories are matched"""
    os.mkdir('dir1')
    os.mkdir('dir2')
    self.wt.add(['dir1', 'dir2'])
    self.wt.rename_one('file1', 'dir1/file1')
    old_tree = self.b.repository.revision_tree(b'rev-1')
    new_tree = self.b.repository.revision_tree(b'rev-4')
    out = get_diff_as_string(old_tree, new_tree, specific_files=['dir1'], working_tree=self.wt)
    self.assertContainsRe(out, b'file1\t')
    out = get_diff_as_string(old_tree, new_tree, specific_files=['dir2'], working_tree=self.wt)
    self.assertNotContainsRe(out, b'file1\t')