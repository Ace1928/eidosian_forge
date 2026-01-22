from . import errors, osutils
def _lookup_by_file_id(self, extra_entries, other_tree, file_id):
    """Lookup an inventory entry by file_id.

        This is called when an entry is missing in the normal order.
        Generally this is because a file was either renamed, or it was
        deleted/added. If the entry was found in the inventory and not in
        extra_entries, it will be added to self._out_of_order_processed

        :param extra_entries: A dictionary of {file_id: (path, ie)}.  This
            should be filled with entries that were found before they were
            used. If file_id is present, it will be removed from the
            dictionary.
        :param other_tree: The Tree to search, in case we didn't find the entry
            yet.
        :param file_id: The file_id to look for
        :return: (path, ie) if found or (None, None) if not present.
        """
    if file_id in extra_entries:
        return extra_entries.pop(file_id)
    try:
        cur_path = other_tree.id2path(file_id)
    except errors.NoSuchId:
        cur_path = None
    if cur_path is None:
        return (None, None)
    else:
        self._out_of_order_processed.add(file_id)
        cur_ie = next(other_tree.iter_entries_by_dir(specific_files=[cur_path]))[1]
        return (cur_path, cur_ie)