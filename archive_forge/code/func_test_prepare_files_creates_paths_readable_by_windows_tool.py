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
def test_prepare_files_creates_paths_readable_by_windows_tool(self):
    self.requireFeature(features.AttribFeature)
    output = BytesIO()
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', b'content')])
    tree.add('file', b'file-id')
    tree.commit('old tree')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    basis_tree = tree.basis_tree()
    basis_tree.lock_read()
    self.addCleanup(basis_tree.unlock)
    diff_obj = diff.DiffFromTool([sys.executable, '-c', 'print "{old_path} {new_path}"'], basis_tree, tree, output)
    diff_obj._prepare_files('file', 'file', file_id=b'file-id')
    self.assertReadableByAttrib(diff_obj._root, 'old\\file', 'R.*old\\\\file$')
    self.assertEndsWith(tree.basedir, 'work/tree')
    self.assertReadableByAttrib(tree.basedir, 'file', 'work\\\\tree\\\\file$')