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
def test_renamed_and_modified_file(self):
    """Test when a file is only renamed."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', b'contents\n')])
    tree.add(['file'], ids=[b'file-id'])
    tree.commit('one', rev_id=b'rev-1')
    tree.rename_one('file', 'newname')
    self.build_tree_contents([('tree/newname', b'new contents\n')])
    d = get_diff_as_string(tree.basis_tree(), tree)
    self.assertContainsRe(d, b"=== renamed file 'file' => 'newname'\n")
    self.assertContainsRe(d, b'--- old/file\t')
    self.assertContainsRe(d, b'\\+\\+\\+ new/newname\t')
    self.assertContainsRe(d, b'-contents\n\\+new contents\n')