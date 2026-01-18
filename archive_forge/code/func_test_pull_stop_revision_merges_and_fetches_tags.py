from breezy import branch, controldir, errors, revision
from breezy.tests import TestNotApplicable, fixtures, per_branch
def test_pull_stop_revision_merges_and_fetches_tags(self):
    """br.pull(source, stop_revision=REV) updates and fetches tags."""
    try:
        builder = self.make_branch_builder('source')
    except errors.UninitializableFormat:
        raise TestNotApplicable('uninitializeable format')
    source, rev1, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
    target = source.controldir.sprout('target').open_branch()
    rev_2_again = builder.build_commit(message='Rev 2 again')
    try:
        source.tags.set_tag('tag-a', rev2)
    except errors.TagsNotSupported:
        raise TestNotApplicable('format does not support tags.')
    source.get_config_stack().set('branch.fetch_tags', True)
    target.pull(source, stop_revision=rev_2_again)
    self.assertEqual(rev2, target.tags.lookup_tag('tag-a'))
    target.repository.get_revision(rev2)