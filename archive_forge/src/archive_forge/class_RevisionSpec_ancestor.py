from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_ancestor(RevisionSpec):
    """Selects a common ancestor with a second branch."""
    help_txt = "Selects a common ancestor with a second branch.\n\n    Supply the path to a branch to select the common ancestor.\n\n    The common ancestor is the last revision that existed in both\n    branches. Usually this is the branch point, but it could also be\n    a revision that was merged.\n\n    This is frequently used with 'diff' to return all of the changes\n    that your branch introduces, while excluding the changes that you\n    have not merged from the remote branch.\n\n    Examples::\n\n      ancestor:/path/to/branch\n      $ bzr diff -r ancestor:../../mainline/branch\n    "
    prefix = 'ancestor:'

    def _match_on(self, branch, revs):
        trace.mutter('matching ancestor: on: %s, %s', self.spec, branch)
        return self._find_revision_info(branch, self.spec)

    def _as_revision_id(self, context_branch):
        return self._find_revision_id(context_branch, self.spec)

    @staticmethod
    def _find_revision_info(branch, other_location):
        revision_id = RevisionSpec_ancestor._find_revision_id(branch, other_location)
        return RevisionInfo(branch, None, revision_id)

    @staticmethod
    def _find_revision_id(branch, other_location):
        from .branch import Branch
        with branch.lock_read():
            revision_a = branch.last_revision()
            if revision_a == revision.NULL_REVISION:
                raise errors.NoCommits(branch)
            if other_location == '':
                other_location = branch.get_parent()
            other_branch = Branch.open(other_location)
            with other_branch.lock_read():
                revision_b = other_branch.last_revision()
                if revision_b == revision.NULL_REVISION:
                    raise errors.NoCommits(other_branch)
                graph = branch.repository.get_graph(other_branch.repository)
                rev_id = graph.find_unique_lca(revision_a, revision_b)
            if rev_id == revision.NULL_REVISION:
                raise errors.NoCommonAncestor(revision_a, revision_b)
            return rev_id