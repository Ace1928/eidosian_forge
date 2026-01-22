from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _copy_item(self, src_path, dest_path, inv):
    newly_changed = self._new_file_ids.get(src_path) or self._modified_file_ids.get(src_path)
    if newly_changed:
        file_id = newly_changed
        ie = self._delta_entries_by_fileid[file_id][3]
    else:
        file_id = inv.path2id(src_path)
        if file_id is None:
            self.warning('ignoring copy of %s to %s - source does not exist', src_path, dest_path)
            return
        ie = inv.get_entry(file_id)
    kind = ie.kind
    if kind == 'file':
        if newly_changed:
            content = self.data_for_commit[file_id]
        else:
            revtree = self.rev_store.repo.revision_tree(self.parents[0])
            content = revtree.get_file_text(src_path)
        self._modify_item(dest_path, kind, ie.executable, content, inv)
    elif kind == 'symlink':
        self._modify_item(dest_path, kind, False, ie.symlink_target.encode('utf-8'), inv)
    else:
        self.warning('ignoring copy of %s %s - feature not yet supported', kind, dest_path)