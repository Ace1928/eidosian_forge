from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _rename_pending_change(self, old_path, new_path, file_id):
    """Instead of adding/modifying old-path, add new-path instead."""
    old_ie = self._delta_entries_by_fileid[file_id][3]
    self.record_delete(old_path, old_ie)
    if old_path in self._new_file_ids:
        del self._new_file_ids[old_path]
    else:
        del self._modified_file_ids[old_path]
    self._new_file_ids[new_path] = file_id
    kind = old_ie.kind
    basename, parent_id = self._ensure_directory(new_path, self.basis_inventory)
    ie = inventory.make_entry(kind, basename, parent_id, file_id)
    ie.revision = self.revision_id
    if kind == 'file':
        ie.executable = old_ie.executable
        ie.text_sha1 = old_ie.text_sha1
        ie.text_size = old_ie.text_size
    elif kind == 'symlink':
        ie.symlink_target = old_ie.symlink_target
    self.record_new(new_path, ie)