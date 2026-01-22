from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_rebase_todo(Command):
    """Print list of revisions that still need to be replayed as part of the
    current rebase operation.

    """
    takes_options = [Option('directory', short_name='d', help='Branch to replay onto, rather than the one containing the working directory.', type=str)]

    def run(self, directory='.'):
        from ...workingtree import WorkingTree
        from .rebase import RebaseState1, rebase_todo
        wt = WorkingTree.open_containing(directory)[0]
        with wt.lock_read():
            state = RebaseState1(wt)
            try:
                replace_map = state.read_plan()[1]
            except NoSuchFile:
                raise CommandError(gettext('No rebase in progress'))
            currentrevid = state.read_active_revid()
            if currentrevid is not None:
                note(gettext('Currently replaying: %s') % currentrevid)
            for revid in rebase_todo(wt.branch.repository, replace_map):
                note(gettext('{0} -> {1}').format(revid, replace_map[revid][0]))