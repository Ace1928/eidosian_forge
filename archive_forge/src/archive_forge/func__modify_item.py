from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _modify_item(self, path, kind, is_executable, data, inv):
    """Add to or change an item in the inventory."""
    existing = self._new_file_ids.get(path)
    if existing:
        if kind != 'directory':
            self.warning('%s already added in this commit - ignoring' % (path,))
        return
    basename, parent_id = self._ensure_directory(path, inv)
    file_id = self.bzr_file_id(path)
    ie = inventory.make_entry(kind, basename, parent_id, file_id)
    ie.revision = self.revision_id
    if kind == 'file':
        ie.executable = is_executable
        ie.text_sha1 = osutils.sha_string(data)
        ie.text_size = len(data)
        self.data_for_commit[file_id] = data
    elif kind == 'directory':
        self.directory_entries[path] = ie
        self.data_for_commit[file_id] = b''
    elif kind == 'symlink':
        ie.symlink_target = self._decode_path(data)
        self.data_for_commit[file_id] = b''
    else:
        self.warning("Cannot import items of kind '%s' yet - ignoring '%s'" % (kind, path))
        return
    try:
        old_ie = inv.get_entry(file_id)
    except errors.NoSuchId:
        try:
            self.record_new(path, ie)
        except:
            print("failed to add path '%s' with entry '%s' in command %s" % (path, ie, self.command.id))
            print("parent's children are:\n%r\n" % (ie.parent_id.children,))
            raise
    else:
        if old_ie.kind == 'directory':
            self.record_delete(path, old_ie)
        self.record_changed(path, ie, parent_id)