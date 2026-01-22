import os
from .file import GitFile
from .index import commit_tree, iter_fresh_objects
from .reflog import drop_reflog_entry, read_reflog
class Stash:
    """A Git stash.

    Note that this doesn't currently update the working tree.
    """

    def __init__(self, repo, ref=DEFAULT_STASH_REF) -> None:
        self._ref = ref
        self._repo = repo

    @property
    def _reflog_path(self):
        return os.path.join(self._repo.commondir(), 'logs', os.fsdecode(self._ref))

    def stashes(self):
        try:
            with GitFile(self._reflog_path, 'rb') as f:
                return reversed(list(read_reflog(f)))
        except FileNotFoundError:
            return []

    @classmethod
    def from_repo(cls, repo):
        """Create a new stash from a Repo object."""
        return cls(repo)

    def drop(self, index):
        """Drop entry with specified index."""
        with open(self._reflog_path, 'rb+') as f:
            drop_reflog_entry(f, index, rewrite=True)
        if len(self) == 0:
            os.remove(self._reflog_path)
            del self._repo.refs[self._ref]
            return
        if index == 0:
            self._repo.refs[self._ref] = self[0].new_sha

    def pop(self, index):
        raise NotImplementedError(self.pop)

    def push(self, committer=None, author=None, message=None):
        """Create a new stash.

        Args:
          committer: Optional committer name to use
          author: Optional author name to use
          message: Optional commit message
        """
        commit_kwargs = {}
        if committer is not None:
            commit_kwargs['committer'] = committer
        if author is not None:
            commit_kwargs['author'] = author
        index = self._repo.open_index()
        index_tree_id = index.commit(self._repo.object_store)
        index_commit_id = self._repo.do_commit(ref=None, tree=index_tree_id, message=b'Index stash', merge_heads=[self._repo.head()], no_verify=True, **commit_kwargs)
        stash_tree_id = commit_tree(self._repo.object_store, iter_fresh_objects(index, os.fsencode(self._repo.path), object_store=self._repo.object_store))
        if message is None:
            message = b'A stash on ' + self._repo.head()
        self._repo.refs[self._ref] = self._repo.head()
        cid = self._repo.do_commit(ref=self._ref, tree=stash_tree_id, message=message, merge_heads=[index_commit_id], no_verify=True, **commit_kwargs)
        return cid

    def __getitem__(self, index):
        return list(self.stashes())[index]

    def __len__(self) -> int:
        return len(list(self.stashes()))