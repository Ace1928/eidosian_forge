import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_utf8_file_ids_and_revision_ids(self):
    main_wt = self.make_branch_and_tree('main')
    main_branch = main_wt.branch
    self.build_tree(['main/a'])
    file_id = 'a-fíle-id'.encode()
    main_wt.add(['a'], ids=[file_id])
    revision_id = 'rév-a'.encode()
    try:
        main_wt.commit('a', rev_id=revision_id)
    except errors.NonAsciiRevisionId:
        raise tests.TestSkipped('non-ascii revision ids not supported by %s' % self.repository_format)
    repo = main_wt.branch.repository
    repo.lock_read()
    self.addCleanup(repo.unlock)
    file_ids = repo.fileids_altered_by_revision_ids([revision_id])
    root_id = main_wt.basis_tree().path2id('')
    if root_id in file_ids:
        self.assertEqual({file_id: {revision_id}, root_id: {revision_id}}, file_ids)
    else:
        self.assertEqual({file_id: {revision_id}}, file_ids)