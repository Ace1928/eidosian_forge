from . import errors, osutils
def _lookup_by_master_path(self, extra_entries, other_tree, master_path):
    return self._lookup_by_file_id(extra_entries, other_tree, self._master_tree.path2id(master_path))