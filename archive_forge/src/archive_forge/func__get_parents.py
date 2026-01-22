from dulwich.object_store import tree_lookup_path
from .. import osutils
from ..bzr.versionedfile import UnavailableRepresentation
from ..errors import NoSuchRevision
from ..graph import Graph
from ..revision import NULL_REVISION
from .mapping import decode_git_path, encode_git_path
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