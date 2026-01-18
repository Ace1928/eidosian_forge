import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_verbose_commit_moved(self):
    wt = self.prepare_simple_history()
    os.mkdir('subdir')
    wt.add(['subdir'])
    wt.rename_one('hello.txt', 'subdir/hello.txt')
    out, err = self.run_bzr('commit -m renamed')
    self.assertEqual('', out)
    self.assertEqual({'Committing to: %s/' % osutils.getcwd(), 'added subdir', 'renamed hello.txt => subdir/hello.txt', 'Committed revision 2.', ''}, set(err.split('\n')))