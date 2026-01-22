from ... import errors
from ...bzr.vf_search import PendingAncestryResult
from ...commands import Command
from ...controldir import ControlDir
from ...option import Option
from ...repository import WriteGroup
from ...revision import NULL_REVISION
class cmd_fix_missing_keys_for_stacking(Command):
    """Fix missing keys for stacking.

    This is the fixer script for <https://bugs.launchpad.net/bzr/+bug/354036>.
    """
    hidden = True
    takes_args = ['branch_url']
    takes_options = [Option('dry-run', help="Show what would be done, but don't actually do anything.")]

    def run(self, branch_url, dry_run=False):
        try:
            bd = ControlDir.open(branch_url)
            b = bd.open_branch(ignore_fallbacks=True)
        except (errors.NotBranchError, errors.InvalidURL):
            raise errors.CommandError('Not a branch or invalid URL: %s' % branch_url)
        b.lock_read()
        try:
            url = b.get_stacked_on_url()
        except (errors.UnstackableRepositoryFormat, errors.NotStacked, errors.UnstackableBranchFormat):
            b.unlock()
            raise errors.CommandError('Not stacked: %s' % branch_url)
        raw_r = b.repository.controldir.open_repository()
        if dry_run:
            raw_r.lock_read()
        else:
            b.unlock()
            b = b.controldir.open_branch()
            b.lock_read()
            raw_r.lock_write()
        try:
            revs = raw_r.all_revision_ids()
            rev_parents = raw_r.get_graph().get_parent_map(revs)
            needed = set()
            map(needed.update, rev_parents.values())
            needed.discard(NULL_REVISION)
            needed = {(rev,) for rev in needed}
            needed = needed - raw_r.inventories.keys()
            if not needed:
                return
            self.outf.write('Missing inventories: %r\n' % needed)
            if dry_run:
                return
            assert raw_r._format.network_name() == b.repository._format.network_name()
            stream = b.repository.inventories.get_record_stream(needed, 'topological', True)
            with WriteGroup(raw_r):
                raw_r.inventories.insert_record_stream(stream)
        finally:
            raw_r.unlock()
        b.unlock()
        self.outf.write('Fixed: %s\n' % branch_url)