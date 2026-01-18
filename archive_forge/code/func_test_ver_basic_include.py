import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_ver_basic_include(self):
    """(versioned) Ensure that -I flag is respected.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.aa')
    self._mk_versioned_file('file0.bb')
    self._mk_versioned_file('file0.cc')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'line1'])
    self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 4)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'line1$'])
    self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '-I', '*.aa', '-I', '*.bb', 'line1'])
    self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 4)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '-I', '*.aa', '-I', '*.bb', 'line1$'])
    self.assertContainsRe(out, 'file0.aa~.:line1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file0.bb~.:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)