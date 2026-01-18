import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_verbose_commit_modified(self):
    wt = self.prepare_simple_history()
    self.build_tree_contents([('hello.txt', b'new contents')])
    out, err = self.run_bzr('commit -m modified')
    self.assertEqual('', out)
    self.assertContainsRe(err, '^Committing to: .*\nmodified hello\\.txt\nCommitted revision 2\\.\n$')