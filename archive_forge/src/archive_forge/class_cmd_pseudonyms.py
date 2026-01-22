from ...commands import Command, display_command
from ...errors import (CommandError, ConflictsInTree, NoWorkingTree,
from ...i18n import gettext
from ...option import Option
from ...trace import note
from ...transport import NoSuchFile
class cmd_pseudonyms(Command):
    """Show a list of 'pseudonym' revisions.

    Pseudonym revisions are revisions that are roughly the same revision,
    usually because they were converted from the same revision in a
    foreign version control system.
    """
    takes_args = ['repository?']
    hidden = True

    def run(self, repository=None):
        from ...controldir import ControlDir
        dir, _ = ControlDir.open_containing(repository)
        r = dir.find_repository()
        from .pseudonyms import find_pseudonyms
        for pseudonyms in find_pseudonyms(r, r.all_revision_ids()):
            self.outf.write(', '.join(pseudonyms) + '\n')