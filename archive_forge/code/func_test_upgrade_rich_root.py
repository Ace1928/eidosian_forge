from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_upgrade_rich_root(self):
    tree = self.make_branch_and_tree('tree', format='rich-root')
    tree.commit('first post')
    upgrade.upgrade('tree')