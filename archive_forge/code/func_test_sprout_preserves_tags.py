import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_preserves_tags(self):
    """Sprout preserves tags, even tags of absent revisions."""
    try:
        builder = self.make_branch_builder('source')
    except errors.UninitializableFormat:
        raise tests.TestSkipped('Uninitializable branch format')
    builder.build_commit(message='Rev 1')
    source = builder.get_branch()
    try:
        source.tags.set_tag('tag-a', b'missing-rev')
    except (errors.TagsNotSupported, errors.GhostTagsNotSupported):
        raise tests.TestNotApplicable('Branch format does not support tags or tags to ghosts.')
    target_bzrdir = self.make_repository('target').controldir
    new_branch = source.sprout(target_bzrdir)
    self.assertEqual(b'missing-rev', new_branch.tags.lookup_tag('tag-a'))