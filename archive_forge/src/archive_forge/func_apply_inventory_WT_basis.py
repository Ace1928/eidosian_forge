from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def apply_inventory_WT_basis(test, basis, delta, invalid_delta=True):
    """Apply delta to basis and return the result.

    This sets the parent and then calls update_basis_by_delta.
    It also puts the basis in the repository under both 'basis' and 'result' to
    allow safety checks made by the WT to succeed, and finally ensures that all
    items in the delta with a new path are present in the WT before calling
    update_basis_by_delta.

    :param basis: An inventory to be used as the basis.
    :param delta: The inventory delta to apply:
    :return: An inventory resulting from the application.
    """
    control = test.make_controldir('tree', format=test.format._matchingcontroldir)
    control.create_repository()
    control.create_branch()
    tree = test.format.initialize(control)
    tree.lock_write()
    try:
        target_entries = _create_repo_revisions(tree.branch.repository, basis, delta, invalid_delta)
        tree._write_inventory(basis)
        tree.set_parent_ids([b'basis'])
    finally:
        tree.unlock()
    with tree.lock_write():
        tree.update_basis_by_delta(b'result', delta)
        if not invalid_delta:
            tree._validate()
    tree = tree.controldir.open_workingtree()
    basis_tree = tree.basis_tree()
    basis_tree.lock_read()
    test.addCleanup(basis_tree.unlock)
    basis_inv = basis_tree.root_inventory
    if target_entries:
        basis_entries = list(basis_inv.iter_entries_by_dir())
        test.assertEqual(target_entries, basis_entries)
    return basis_inv