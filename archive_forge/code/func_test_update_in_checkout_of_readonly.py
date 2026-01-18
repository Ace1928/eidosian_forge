from breezy import branch, errors
from breezy import revision as _mod_revision
from breezy import tests
from breezy.tests import per_branch
def test_update_in_checkout_of_readonly(self):
    tree1 = self.make_branch_and_tree('tree1')
    rev1 = tree1.commit('one')
    try:
        tree1.branch.tags.set_tag('test-tag', rev1)
    except errors.TagsNotSupported:
        raise tests.TestNotApplicable('only triggered from branches with tags')
    readonly_branch1 = branch.Branch.open('readonly+' + tree1.branch.base)
    tree2 = tree1.controldir.sprout('tree2').open_workingtree()
    try:
        tree2.branch.bind(readonly_branch1)
    except branch.BindingUnsupported:
        raise tests.TestNotApplicable('only triggered in bound branches')
    rev2 = tree1.commit('two')
    tree2.update()
    self.assertEqual(rev2, tree2.branch.last_revision())