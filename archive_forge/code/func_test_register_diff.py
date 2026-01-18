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
def test_register_diff(self):
    self.create_old_new()
    old_diff_factories = diff.DiffTree.diff_factories
    diff.DiffTree.diff_factories = old_diff_factories[:]
    diff.DiffTree.diff_factories.insert(0, DiffWasIs.from_diff_tree)
    try:
        differ = diff.DiffTree(self.old_tree, self.new_tree, BytesIO())
    finally:
        diff.DiffTree.diff_factories = old_diff_factories
    differ.diff('olddir/oldfile', 'newdir/newfile')
    self.assertNotContainsRe(differ.to_file.getvalue(), b'--- olddir/oldfile.*\\n\\+\\+\\+ newdir/newfile.*\\n\\@\\@ -1,1 \\+1,1 \\@\\@\\n-old\\n\\+new\\n\\n')
    self.assertContainsRe(differ.to_file.getvalue(), b'was: old\nis: new\n')