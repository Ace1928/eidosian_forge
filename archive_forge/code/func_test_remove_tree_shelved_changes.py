import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_shelved_changes(self):
    tree = self.make_branch_and_tree('.')
    creator = shelf.ShelfCreator(tree, tree.basis_tree(), [])
    self.addCleanup(creator.finalize)
    shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
    output = self.run_bzr_error(['Working tree .* has shelved changes'], 'remove-tree', retcode=3)