from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_pull_delta_when_stacked(self):
    if not self.branch_format.supports_stacking():
        raise TestNotApplicable('%r does not support stacking' % self.branch_format)
    stack_on = self.make_branch_and_tree('stack-on')
    text_lines = [b'line %d blah blah blah\n' % i for i in range(20)]
    self.build_tree_contents([('stack-on/a', b''.join(text_lines))])
    stack_on.add('a')
    stack_on.commit('base commit')
    stacked_dir = stack_on.controldir.sprout('stacked', stacked=True)
    stacked_tree = stacked_dir.open_workingtree()
    other_dir = stack_on.controldir.sprout('other')
    other_tree = other_dir.open_workingtree()
    text_lines[9] = b'changed in other\n'
    self.build_tree_contents([('other/a', b''.join(text_lines))])
    stacked_revid = other_tree.commit('commit in other')
    stacked_tree.pull(other_tree.branch)
    stacked_tree.branch.repository.pack()
    check.check_dwim(stacked_tree.branch.base, False, True, True)
    self.check_lines_added_or_present(stacked_tree.branch, stacked_revid)