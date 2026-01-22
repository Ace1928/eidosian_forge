from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def _create_repo_revisions(repo, basis, delta, invalid_delta):
    with repository.WriteGroup(repo):
        rev = revision.Revision(b'basis', timestamp=0, timezone=None, message='', committer='foo@example.com')
        basis.revision_id = b'basis'
        create_texts_for_inv(repo, basis)
        repo.add_revision(b'basis', rev, basis)
        if invalid_delta:
            result_inv = basis
            result_inv.revision_id = b'result'
            target_entries = None
        else:
            result_inv = basis.create_by_apply_delta(delta, b'result')
            create_texts_for_inv(repo, result_inv)
            target_entries = list(result_inv.iter_entries_by_dir())
        rev = revision.Revision(b'result', timestamp=0, timezone=None, message='', committer='foo@example.com')
        repo.add_revision(b'result', rev, result_inv)
    return target_entries