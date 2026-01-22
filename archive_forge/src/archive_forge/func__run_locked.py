import contextlib
from .branch import Branch
from .errors import CommandError, NoSuchRevision
from .trace import note
def _run_locked(self):
    installed = []
    failed = []
    if self.this_branch.last_revision() is None:
        print('No revisions in branch.')
        return
    ghosts = self.iter_ghosts()
    for revision in ghosts:
        try:
            self.this_branch.fetch(self.other_branch, revision)
            installed.append(revision)
        except NoSuchRevision:
            failed.append(revision)
    return (installed, failed)