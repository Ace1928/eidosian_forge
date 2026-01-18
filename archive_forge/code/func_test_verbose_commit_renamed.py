import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_verbose_commit_renamed(self):
    wt = self.prepare_simple_history()
    wt.rename_one('hello.txt', 'gutentag.txt')
    out, err = self.run_bzr('commit -m renamed')
    self.assertEqual('', out)
    self.assertContainsRe(err, '^Committing to: .*\nrenamed hello\\.txt => gutentag\\.txt\nCommitted revision 2\\.$\n')