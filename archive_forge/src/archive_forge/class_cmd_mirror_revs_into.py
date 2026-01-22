from ... import errors
from ...bzr.vf_search import PendingAncestryResult
from ...commands import Command
from ...controldir import ControlDir
from ...option import Option
from ...repository import WriteGroup
from ...revision import NULL_REVISION
class cmd_mirror_revs_into(Command):
    """Mirror all revs from one repo into another."""
    hidden = True
    takes_args = ['source', 'destination']
    _see_also = ['fetch-all-records']

    def run(self, source, destination):
        bd = ControlDir.open(source)
        source_r = bd.open_branch().repository
        bd = ControlDir.open(destination)
        target_r = bd.open_branch().repository
        with source_r.lock_read(), target_r.lock_write():
            revs = [k[-1] for k in source_r.revisions.keys()]
            target_r.fetch(source_r, fetch_spec=PendingAncestryResult(revs, source_r))