import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_rm(self):
    wt = self._make_mixed_case_tree()
    self.run_bzr('add')
    self.run_bzr('ci -m message')
    got = self.run_bzr('rm camelcaseparent LOWERCASEPARENT')[1]
    for expected in ['lowercaseparent/lowercase', 'CamelCaseParent/CamelCase']:
        self.assertContainsRe(got, 'deleted ' + expected + '\n')