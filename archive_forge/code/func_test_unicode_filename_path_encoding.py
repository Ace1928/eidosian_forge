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
def test_unicode_filename_path_encoding(self):
    """Test for bug #382699: unicode filenames on Windows should be shown
        in user encoding.
        """
    self.requireFeature(features.UnicodeFilenameFeature)
    _russian_test = 'Тест'
    directory = _russian_test + '/'
    test_txt = _russian_test + '.txt'
    u1234 = 'ሴ.txt'
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([(test_txt, b'foo\n'), (u1234, b'foo\n'), (directory, None)])
    tree.add([test_txt, u1234, directory])
    sio = BytesIO()
    diff.show_diff_trees(tree.basis_tree(), tree, sio, path_encoding='cp1251')
    output = subst_dates(sio.getvalue())
    shouldbe = b"=== added directory '%(directory)s'\n=== added file '%(test_txt)s'\n--- a/%(test_txt)s\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ b/%(test_txt)s\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -0,0 +1,1 @@\n+foo\n\n=== added file '?.txt'\n--- a/?.txt\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ b/?.txt\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -0,0 +1,1 @@\n+foo\n\n" % {b'directory': _russian_test.encode('cp1251'), b'test_txt': test_txt.encode('cp1251')}
    self.assertEqualDiff(output, shouldbe)