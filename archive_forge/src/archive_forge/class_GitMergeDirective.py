import time
from io import BytesIO
from dulwich import __version__ as dulwich_version
from dulwich.objects import Blob
from .. import __version__ as brz_version
from .. import branch as _mod_branch
from .. import diff as _mod_diff
from .. import errors, osutils
from .. import revision as _mod_revision
from ..merge_directive import BaseMergeDirective
from .mapping import object_mode
from .object_store import get_object_store
class GitMergeDirective(BaseMergeDirective):
    multiple_output_files = True

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, source_branch=None, message=None, patches=None, local_target_branch=None):
        super().__init__(revision_id=revision_id, testament_sha1=testament_sha1, time=time, timezone=timezone, target_branch=target_branch, patch=None, source_branch=source_branch, message=message, bundle=None)
        self.patches = patches

    def to_lines(self):
        return self.patch.splitlines(True)

    def to_files(self):
        return ((summary, patch.splitlines(True)) for summary, patch in self.patches)

    @classmethod
    def _generate_commit(cls, repository, revision_id, num, total, context=_mod_diff.DEFAULT_CONTEXT_AMOUNT):
        s = BytesIO()
        store = get_object_store(repository)
        with store.lock_read():
            commit = store[repository.lookup_bzr_revision_id(revision_id)[0]]
        from dulwich.patch import get_summary, write_commit_patch
        try:
            lhs_parent = repository.get_revision(revision_id).parent_ids[0]
        except IndexError:
            lhs_parent = _mod_revision.NULL_REVISION
        tree_1 = repository.revision_tree(lhs_parent)
        tree_2 = repository.revision_tree(revision_id)
        contents = BytesIO()
        differ = GitDiffTree.from_trees_options(tree_1, tree_2, contents, 'utf8', None, 'a/', 'b/', None, context_lines=context)
        differ.show_diff(None, None)
        write_commit_patch(s, commit, contents.getvalue(), (num, total), version_tail)
        summary = generate_patch_filename(num, get_summary(commit))
        return (summary, s.getvalue())

    @classmethod
    def from_objects(cls, repository, revision_id, time, timezone, target_branch, local_target_branch=None, public_branch=None, message=None):
        patches = []
        submit_branch = _mod_branch.Branch.open(target_branch)
        with submit_branch.lock_read():
            submit_revision_id = submit_branch.last_revision()
            repository.fetch(submit_branch.repository, submit_revision_id)
            graph = repository.get_graph()
            todo = graph.find_difference(submit_revision_id, revision_id)[1]
            total = len(todo)
            for i, revid in enumerate(graph.iter_topo_order(todo)):
                patches.append(cls._generate_commit(repository, revid, i + 1, total))
        return cls(revision_id, None, time, timezone, target_branch=target_branch, source_branch=public_branch, message=message, patches=patches)