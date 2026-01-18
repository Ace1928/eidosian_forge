import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_tags(self):
    """Tags are updated by pull, and revisions named in those tags are
        fetched.
        """
    builder = self.make_branch_builder('source')
    source, rev1, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
    source.get_config_stack().set('branch.fetch_tags', True)
    target_bzrdir = source.controldir.sprout('target')
    source.tags.set_tag('tag-a', rev2)
    self.run_bzr('pull -d target source')
    target = target_bzrdir.open_branch()
    self.assertEqual(rev2, target.tags.lookup_tag('tag-a'))
    target.repository.get_revision(rev2)