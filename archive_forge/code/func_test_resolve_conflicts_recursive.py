import os
from ... import tests
from ...conflicts import resolve
from ...tests import scenarios
from ...tests.test_conflicts import vary_by_conflicts
from .. import conflicts as bzr_conflicts
def test_resolve_conflicts_recursive(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir/', 'dir/hello'])
    tree.add(['dir', 'dir/hello'])
    dirhello = [bzr_conflicts.TextConflict('dir/hello')]
    tree.set_conflicts(dirhello)
    resolve(tree, ['dir'], recursive=False, ignore_misses=True)
    self.assertEqual(dirhello, tree.conflicts())
    resolve(tree, ['dir'], recursive=True, ignore_misses=True)
    self.assertEqual(bzr_conflicts.ConflictList([]), tree.conflicts())