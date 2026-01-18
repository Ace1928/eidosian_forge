import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_upgrade_preserves_signatures(self):
    if not self.repository_format.supports_revision_signatures:
        raise tests.TestNotApplicable('repository does not support signing revisions')
    wt = self.make_branch_and_tree('source')
    a = wt.commit('A', allow_pointless=True)
    repo = wt.branch.repository
    repo.lock_write()
    repo.start_write_group()
    try:
        repo.sign_revision(a, gpg.LoopbackGPGStrategy(None))
    except errors.UnsupportedOperation:
        self.assertFalse(repo._format.supports_revision_signatures)
        raise tests.TestNotApplicable('signatures not supported by repository format')
    repo.commit_write_group()
    repo.unlock()
    old_signature = repo.get_signature_text(a)
    try:
        old_format = controldir.ControlDirFormat.get_default_format()
        format = controldir.format_registry.make_controldir('development-subtree')
        upgrade.upgrade(repo.controldir.root_transport.base, format=format)
    except errors.UpToDateFormat:
        return
    except errors.BadConversionTarget as e:
        raise tests.TestSkipped(str(e))
    wt = workingtree.WorkingTree.open(wt.basedir)
    new_signature = wt.branch.repository.get_signature_text(a)
    self.assertEqual(old_signature, new_signature)