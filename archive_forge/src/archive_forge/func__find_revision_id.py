from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
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