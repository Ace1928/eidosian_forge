from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
def upgrade_repository(repository, generate_rebase_map, determine_new_revid, revision_id=None, allow_changes=False, verbose=False):
    """Upgrade the revisions in repository until the specified stop revision.

    :param repository: Repository in which to upgrade.
    :param foreign_repository: Repository to fetch new revisions from.
    :param new_mapping: New mapping.
    :param revision_id: Revision id up until which to upgrade, or None for
                        all revisions.
    :param allow_changes: Allow changes to mappings.
    :param verbose: Whether to print list of rewrites
    :return: Dictionary of mapped revisions
    """
    with repository.lock_write():
        plan, revid_renames = create_upgrade_plan(repository, generate_rebase_map, determine_new_revid, revision_id=revision_id, allow_changes=allow_changes)
        if verbose:
            for revid in rebase_todo(repository, plan):
                trace.note('{} -> {}'.format(revid, plan[revid][0]))
        rebase(repository, plan, CommitBuilderRevisionRewriter(repository))
        return revid_renames