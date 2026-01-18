from . import errors
from . import revision as _mod_revision
from .branch import Branch
from .errors import BoundBranchOutOfDate
Remove the last revision from the supplied branch.

    :param dry_run: Don't actually change anything
    :param verbose: Print each step as you do it
    :param revno: Remove back to this revision
    :param local: If this branch is bound, only remove the revisions from the
        local branch. If this branch is not bound, it is an error to pass
        local=True.
    :param keep_tags: Whether to keep tags pointing at the removed revisions
        around.
    