import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_unicode_only_file(self):
    """Test filename and contents that requires a unicode encoding"""
    tree = self.make_branch_and_tree('.')
    contents = ['ሴ']
    self.build_tree(contents)
    tree.add(contents)
    tree.commit('Initial commit')
    as_utf8 = 'ሴ'
    streams = self.run_bzr_raw(['grep', '--files-with-matches', 'contents'], encoding='UTF-8')
    as_utf8 = as_utf8.encode('UTF-8')
    self.assertEqual(streams, (as_utf8 + b'\n', b''))
    streams = self.run_bzr_raw(['grep', '-r', '1', '--files-with-matches', 'contents'], encoding='UTF-8')
    self.assertEqual(streams, (as_utf8 + b'~1\n', b''))
    fileencoding = osutils.get_user_encoding()
    as_mangled = as_utf8.decode(fileencoding, 'replace').encode('UTF-8')
    streams = self.run_bzr_raw(['grep', '-n', 'contents'], encoding='UTF-8')
    self.assertEqual(streams, (b'%s:1:contents of %s\n' % (as_utf8, as_mangled), b''))
    streams = self.run_bzr_raw(['grep', '-n', '-r', '1', 'contents'], encoding='UTF-8')
    self.assertEqual(streams, (b'%s~1:1:contents of %s\n' % (as_utf8, as_mangled), b''))