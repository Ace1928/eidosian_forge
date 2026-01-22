import os
import dulwich
from dulwich.repo import Repo as GitRepo
from ... import config, errors, revision
from ...repository import InterRepository, Repository
from .. import dir, repository, tests
from ..mapping import default_mapping
from ..object_store import BazaarObjectStore
from ..push import MissingObjectsIterator
class SigningGitRepository(tests.TestCaseWithTransport):

    def test_signed_commit(self):
        import breezy.gpg
        oldstrategy = breezy.gpg.GPGStrategy
        wt = self.make_branch_and_tree('.', format='git')
        branch = wt.branch
        revid = wt.commit('base', allow_pointless=True)
        self.assertFalse(branch.repository.has_signature_for_revision_id(revid))
        try:
            breezy.gpg.GPGStrategy = breezy.gpg.LoopbackGPGStrategy
            conf = config.MemoryStack(b'\ncreate_signatures=always\n')
            revid2 = wt.commit(config=conf, message='base', allow_pointless=True)

            def sign(text):
                return breezy.gpg.LoopbackGPGStrategy(None).sign(text)
            self.assertIsInstance(branch.repository.get_signature_text(revid2), bytes)
        finally:
            breezy.gpg.GPGStrategy = oldstrategy