from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _rename_item(self, old_path, new_path, inv):
    existing = self._new_file_ids.get(old_path) or self._modified_file_ids.get(old_path)
    if existing:
        self._rename_pending_change(old_path, new_path, existing)
        return
    file_id = inv.path2id(old_path)
    if file_id is None:
        self.warning('ignoring rename of %s to %s - old path does not exist' % (old_path, new_path))
        return
    ie = inv.get_entry(file_id)
    rev_id = ie.revision
    new_file_id = inv.path2id(new_path)
    if new_file_id is not None:
        self.record_delete(new_path, inv.get_entry(new_file_id))
    self.record_rename(old_path, new_path, file_id, ie)
    lines = self.rev_store.get_file_lines(rev_id, old_path)
    self.data_for_commit[file_id] = b''.join(lines)