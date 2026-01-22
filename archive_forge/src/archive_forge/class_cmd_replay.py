from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_replay(Command):
    """Replay commits from another branch on top of this one.

    """
    takes_options = ['revision', 'merge-type', Option('directory', short_name='d', help='Branch to replay onto, rather than the one containing the working directory.', type=str)]
    takes_args = ['location']
    hidden = True

    def run(self, location, revision=None, merge_type=None, directory='.'):
        from ... import ui
        from ...branch import Branch
        from ...workingtree import WorkingTree
        from .rebase import RebaseState1, WorkingTreeRevisionRewriter, regenerate_default_revid
        from_branch = Branch.open_containing(location)[0]
        if revision is not None:
            if len(revision) == 1:
                if revision[0] is not None:
                    todo = [revision[0].as_revision_id(from_branch)]
            elif len(revision) == 2:
                from_revno, from_revid = revision[0].in_history(from_branch)
                to_revno, to_revid = revision[1].in_history(from_branch)
                if to_revid is None:
                    to_revno = from_branch.revno()
                todo = []
                for revno in range(from_revno, to_revno + 1):
                    todo.append(from_branch.get_rev_id(revno))
            else:
                raise CommandError(gettext('--revision takes only one or two arguments'))
        else:
            raise CommandError(gettext('--revision is mandatory'))
        wt = WorkingTree.open(directory)
        wt.lock_write()
        try:
            state = RebaseState1(wt)
            replayer = WorkingTreeRevisionRewriter(wt, state, merge_type=merge_type)
            pb = ui.ui_factory.nested_progress_bar()
            try:
                for revid in todo:
                    pb.update(gettext('replaying commits'), todo.index(revid), len(todo))
                    wt.branch.repository.fetch(from_branch.repository, revid)
                    newrevid = regenerate_default_revid(wt.branch.repository, revid)
                    replayer(revid, newrevid, [wt.last_revision()])
            finally:
                pb.finished()
        finally:
            wt.unlock()