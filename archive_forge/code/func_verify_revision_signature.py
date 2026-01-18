from io import BytesIO
from dulwich.errors import NotCommitError
from dulwich.object_store import peel_sha, tree_lookup_path
from dulwich.objects import ZERO_SHA, Commit
from .. import check, errors
from .. import graph as _mod_graph
from .. import lock, repository
from .. import revision as _mod_revision
from .. import trace, transactions, ui
from ..decorators import only_raises
from ..foreign import ForeignRepository
from .filegraph import GitFileLastChangeScanner, GitFileParentProvider
from .mapping import (default_mapping, encode_git_path, foreign_vcs_git,
from .tree import GitRevisionTree
def verify_revision_signature(self, revision_id, gpg_strategy):
    """Verify the signature on a revision.

        :param revision_id: the revision to verify
        :gpg_strategy: the GPGStrategy object to used

        :return: gpg.SIGNATURE_VALID or a failed SIGNATURE_ value
        """
    from breezy import gpg
    with self.lock_read():
        git_commit_id, mapping = self.lookup_bzr_revision_id(revision_id)
        try:
            commit = self._git.object_store[git_commit_id]
        except KeyError:
            raise errors.NoSuchRevision(self, revision_id)
        if commit.gpgsig is None:
            return (gpg.SIGNATURE_NOT_SIGNED, None)
        without_sig = Commit.from_string(commit.as_raw_string())
        without_sig.gpgsig = None
        result, key, plain_text = gpg_strategy.verify(without_sig.as_raw_string(), commit.gpgsig)
        return (result, key)