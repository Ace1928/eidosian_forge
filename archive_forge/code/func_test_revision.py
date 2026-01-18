import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def test_revision(self):
    tree = self.create_tree()
    branch = self.make_branch('just_branch')
    branch.pull(tree.branch)
    txt = self.run_bzr('version-info -r1 just_branch')[0]
    self.assertStartsWith(txt, 'revision-id: r1\n')