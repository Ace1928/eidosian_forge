from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_unstack_fetches(self):
    """Removing the stacked-on branch pulls across all data"""
    try:
        builder = self.make_branch_builder('trunk')
    except errors.UninitializableFormat:
        raise TestNotApplicable('uninitializeable format')
    trunk, mainline_revid, rev2 = fixtures.build_branch_with_non_ancestral_rev(builder)
    try:
        new_dir = trunk.controldir.sprout(self.get_url('newbranch'), stacked=True)
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    self.assertRevisionNotInRepository('newbranch', mainline_revid)
    new_branch = new_dir.open_branch()
    try:
        new_branch.tags.set_tag('tag-a', rev2)
    except errors.TagsNotSupported:
        tags_supported = False
    else:
        tags_supported = True
    new_branch.set_stacked_on_url(None)
    self.assertRevisionInRepository('newbranch', mainline_revid)
    self.assertRevisionInRepository('trunk', mainline_revid)
    if tags_supported:
        self.assertRevisionInRepository('newbranch', rev2)
    self.assertRaises(errors.NotStacked, new_branch.get_stacked_on_url)