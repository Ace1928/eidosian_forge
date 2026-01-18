import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_ver_null_option(self):
    """(versioned) --null option should use NUL instead of newline.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt', total_lines=3)
    nref = ud.normalize('NFC', 'file0.txt~1:line1\x00file0.txt~1:line2\x00file0.txt~1:line3\x00')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--null', 'line[1-3]'])
    nout = ud.normalize('NFC', out)
    self.assertEqual(nout, nref)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '-Z', 'line[1-3]'])
    nout = ud.normalize('NFC', out)
    self.assertEqual(nout, nref)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--null', 'line'])
    nout = ud.normalize('NFC', out)
    self.assertEqual(nout, nref)
    self.assertEqual(len(out.splitlines()), 1)