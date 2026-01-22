from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
class MissingObjectsIterator:
    """Iterate over git objects that are missing from a target repository.

    """

    def __init__(self, store, source, pb=None):
        """Create a new missing objects iterator.

        """
        self.source = source
        self._object_store = store
        self._pending = []
        self.pb = pb

    def import_revisions(self, revids, lossy):
        """Import a set of revisions into this git repository.

        :param revids: Revision ids of revisions to import
        :param lossy: Whether to not roundtrip bzr metadata
        """
        for i, revid in enumerate(revids):
            if self.pb:
                self.pb.update('pushing revisions', i, len(revids))
            git_commit = self.import_revision(revid, lossy)
            yield (revid, git_commit)

    def import_revision(self, revid, lossy):
        """Import a revision into this Git repository.

        :param revid: Revision id of the revision
        :param roundtrip: Whether to roundtrip bzr metadata
        """
        tree = self._object_store.tree_cache.revision_tree(revid)
        rev = self.source.get_revision(revid)
        commit = None
        for path, obj in self._object_store._revision_to_objects(rev, tree, lossy):
            if obj.type_name == b'commit':
                commit = obj
            self._pending.append((obj, path))
        if commit is None:
            raise AssertionError('no commit object generated for revision %s' % revid)
        return commit.id

    def __len__(self):
        return len(self._pending)

    def __iter__(self):
        return iter(self._pending)