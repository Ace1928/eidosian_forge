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
def test_show_diff_specified(self):
    """A working tree filename can be used to identify a file"""
    self.wt.rename_one('file1', 'file1b')
    old_tree = self.b.repository.revision_tree(b'rev-1')
    new_tree = self.b.repository.revision_tree(b'rev-4')
    out = get_diff_as_string(old_tree, new_tree, specific_files=['file1b'], working_tree=self.wt)
    self.assertContainsRe(out, b'file1\t')