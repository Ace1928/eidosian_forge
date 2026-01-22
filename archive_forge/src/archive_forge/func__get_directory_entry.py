from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _get_directory_entry(self, inv, dirname):
    """Get the inventory entry for a directory.

        Raises KeyError if dirname is not a directory in inv.
        """
    result = self.directory_entries.get(dirname)
    if result is None:
        if dirname in self._paths_deleted_this_commit:
            raise KeyError
        try:
            file_id = inv.path2id(dirname)
        except errors.NoSuchId:
            raise KeyError
        if file_id is None:
            raise KeyError
        result = inv.get_entry(file_id)
        if result.kind == 'directory':
            self.directory_entries[dirname] = result
        else:
            raise KeyError
    return result