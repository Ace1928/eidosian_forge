from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_pull_tag_selector(self):
    if not self.branch_format_from.supports_tags():
        raise TestNotApplicable('from format does not support tags')
    if not self.branch_format_to.supports_tags():
        raise TestNotApplicable('to format does not support tags')
    tree_a = self.make_from_branch_and_tree('tree_a')
    revid1 = tree_a.commit('message 1')
    try:
        tree_b = self.sprout_to(tree_a.controldir, 'tree_b').open_workingtree()
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    tree_b.branch.tags.set_tag('tag1', revid1)
    tree_b.branch.tags.set_tag('tag2', revid1)
    tree_b.branch.get_config_stack().set('branch.fetch_tags', True)
    tree_a.pull(tree_b.branch, tag_selector=lambda x: x == 'tag1')
    self.assertEqual({'tag1': revid1}, tree_a.branch.tags.get_tag_dict())