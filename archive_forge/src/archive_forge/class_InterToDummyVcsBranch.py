from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class InterToDummyVcsBranch(branch.GenericInterBranch):

    @staticmethod
    def is_compatible(source, target):
        return isinstance(target, DummyForeignVcsBranch)

    def push(self, overwrite=False, stop_revision=None, lossy=False, tag_selector=None):
        if not lossy:
            raise errors.NoRoundtrippingSupport(self.source, self.target)
        result = branch.BranchPushResult()
        result.source_branch = self.source
        result.target_branch = self.target
        result.old_revno, result.old_revid = self.target.last_revision_info()
        self.source.lock_read()
        try:
            graph = self.source.repository.get_graph()
            my_history = branch_history(self.target.repository.get_graph(), result.old_revid)
            if stop_revision is None:
                stop_revision = self.source.last_revision()
            their_history = branch_history(graph, stop_revision)
            if their_history[:min(len(my_history), len(their_history))] != my_history:
                raise errors.DivergedBranches(self.target, self.source)
            todo = their_history[len(my_history):]
            revidmap = {}
            for revid in todo:
                rev = self.source.repository.get_revision(revid)
                tree = self.source.repository.revision_tree(revid)

                def get_file_with_stat(path):
                    return (tree.get_file(path), None)
                tree.get_file_with_stat = get_file_with_stat
                new_revid = self.target.mapping.revision_id_foreign_to_bzr((b'%d' % rev.timestamp, str(rev.timezone).encode('ascii'), str(self.target.revno()).encode('ascii')))
                parent_revno, parent_revid = self.target.last_revision_info()
                if parent_revid == revision.NULL_REVISION:
                    parent_revids = []
                else:
                    parent_revids = [parent_revid]
                builder = self.target.get_commit_builder(parent_revids, self.target.get_config_stack(), rev.timestamp, rev.timezone, rev.committer, rev.properties, new_revid)
                try:
                    parent_tree = self.target.repository.revision_tree(parent_revid)
                    iter_changes = tree.iter_changes(parent_tree)
                    list(builder.record_iter_changes(tree, parent_revid, iter_changes))
                    builder.finish_inventory()
                except:
                    builder.abort()
                    raise
                revidmap[revid] = builder.commit(rev.message)
                self.target.set_last_revision_info(parent_revno + 1, revidmap[revid])
                trace.mutter('lossily pushed revision %s -> %s', revid, revidmap[revid])
        finally:
            self.source.unlock()
        result.new_revno, result.new_revid = self.target.last_revision_info()
        result.revidmap = revidmap
        return result