from ... import osutils, trace, ui
from ...errors import BzrError
from .rebase import (CommitBuilderRevisionRewriter, generate_transpose_plan,
def upgrade_tags(tags, repository, generate_rebase_map, determine_new_revid, allow_changes=False, verbose=False, branch_renames=None, branch_ancestry=None):
    """Upgrade a tags dictionary."""
    renames = {}
    if branch_renames is not None:
        renames.update(branch_renames)
    pb = ui.ui_factory.nested_progress_bar()
    try:
        tags_dict = tags.get_tag_dict()
        for i, (name, revid) in enumerate(tags_dict.iteritems()):
            pb.update('upgrading tags', i, len(tags_dict))
            if revid not in renames:
                try:
                    repository.lock_read()
                    revid_exists = repository.has_revision(revid)
                finally:
                    repository.unlock()
                if revid_exists:
                    renames.update(upgrade_repository(repository, generate_rebase_map, determine_new_revid, revision_id=revid, allow_changes=allow_changes, verbose=verbose))
            if revid in renames and (branch_ancestry is None or revid not in branch_ancestry):
                tags.set_tag(name, renames[revid])
    finally:
        pb.finished()