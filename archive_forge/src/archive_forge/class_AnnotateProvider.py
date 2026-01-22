from dulwich.object_store import tree_lookup_path
from .. import osutils
from ..bzr.versionedfile import UnavailableRepresentation
from ..errors import NoSuchRevision
from ..graph import Graph
from ..revision import NULL_REVISION
from .mapping import decode_git_path, encode_git_path
class AnnotateProvider:

    def __init__(self, change_scanner):
        self.change_scanner = change_scanner
        self.store = self.change_scanner.repository._git.object_store

    def _get_parents(self, path, text_revision):
        commit_id, mapping = self.change_scanner.repository.lookup_bzr_revision_id(text_revision)
        text_parents = []
        path = encode_git_path(path)
        for commit_parent in self.store[commit_id].parents:
            try:
                store, path, text_parent = self.change_scanner.find_last_change_revision(path, commit_parent)
            except KeyError:
                continue
            if text_parent not in text_parents:
                text_parents.append(text_parent)
        return tuple([(decode_git_path(path), self.change_scanner.repository.lookup_foreign_revision_id(p)) for p in text_parents])

    def get_parent_map(self, keys):
        ret = {}
        for key in keys:
            path, text_revision = key
            if text_revision == NULL_REVISION:
                ret[key] = ()
                continue
            try:
                ret[key] = self._get_parents(path, text_revision)
            except KeyError:
                pass
        return ret

    def get_record_stream(self, keys, ordering, include_delta_closure):
        if ordering == 'topological':
            graph = Graph(self)
            keys = graph.iter_topo_order(keys)
        store = self.change_scanner.repository._git.object_store
        for path, text_revision in keys:
            try:
                commit_id, mapping = self.change_scanner.repository.lookup_bzr_revision_id(text_revision)
            except NoSuchRevision:
                yield GitAbsentContentFactory(store, path, text_revision)
                continue
            try:
                tree_id = store[commit_id].tree
            except KeyError:
                yield GitAbsentContentFactory(store, path, text_revision)
                continue
            try:
                mode, blob_sha = tree_lookup_path(store.__getitem__, tree_id, encode_git_path(path))
            except KeyError:
                yield GitAbsentContentFactory(store, path, text_revision)
            else:
                yield GitBlobContentFactory(store, path, text_revision, blob_sha)