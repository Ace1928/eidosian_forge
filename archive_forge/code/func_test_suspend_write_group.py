import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_suspend_write_group(self):
    repo = self.make_write_locked_repo()
    repo.start_write_group()
    repo.texts.add_lines((b'file-id', b'revid'), (), [b'lines'])
    try:
        wg_tokens = repo.suspend_write_group()
    except errors.UnsuspendableWriteGroup:
        self.assertTrue(repo.is_in_write_group())
        repo.abort_write_group()
    else:
        self.assertFalse(repo.is_in_write_group())
        self.assertEqual(1, len(wg_tokens))
        self.assertIsInstance(wg_tokens[0], str)