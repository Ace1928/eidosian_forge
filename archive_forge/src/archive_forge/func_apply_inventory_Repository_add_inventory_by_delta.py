from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def apply_inventory_Repository_add_inventory_by_delta(self, basis, delta, invalid_delta=True):
    """Apply delta to basis and return the result.

    This inserts basis as a whole inventory and then uses
    add_inventory_by_delta to add delta.

    :param basis: An inventory to be used as the basis.
    :param delta: The inventory delta to apply:
    :return: An inventory resulting from the application.
    """
    format = self.format()
    control = self.make_controldir('tree', format=format._matchingcontroldir)
    repo = format.initialize(control)
    with repo.lock_write(), repository.WriteGroup(repo):
        rev = revision.Revision(b'basis', timestamp=0, timezone=None, message='', committer='foo@example.com')
        basis.revision_id = b'basis'
        create_texts_for_inv(repo, basis)
        repo.add_revision(b'basis', rev, basis)
    with repo.lock_write(), repository.WriteGroup(repo):
        inv_sha1 = repo.add_inventory_by_delta(b'basis', delta, b'result', [b'basis'])
    repo = repo.controldir.open_repository()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    return repo.get_inventory(b'result')