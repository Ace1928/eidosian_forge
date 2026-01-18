from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
def upgrade_branch(branch, generate_rebase_map, determine_new_revid, allow_changes=False, verbose=False):
    """Upgrade a branch to the current mapping version.

    :param branch: Branch to upgrade.
    :param foreign_repository: Repository to fetch new revisions from
    :param allow_changes: Allow changes in mappings.
    :param verbose: Whether to print verbose list of rewrites
    """
    revid = branch.last_revision()
    renames = upgrade_repository(branch.repository, generate_rebase_map, determine_new_revid, revision_id=revid, allow_changes=allow_changes, verbose=verbose)
    if revid in renames:
        branch.generate_revision_history(renames[revid])
    ancestry = branch.repository.get_ancestry(branch.last_revision(), topo_sorted=False)
    upgrade_tags(branch.tags, branch.repository, generate_rebase_map, determine_new_revid, allow_changes=allow_changes, verbose=verbose, branch_renames=renames, branch_ancestry=ancestry)
    return renames