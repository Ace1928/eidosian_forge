class MapTree:
    """Wrapper around a tree that translates file ids.
    """

    def __init__(self, oldtree, fileid_map):
        """Create a new MapTree.

        :param oldtree: Old tree to map to.
        :param fileid_map: Map with old -> new file ids.
        """
        self.oldtree = oldtree
        self.map = fileid_map

    def old_id(self, file_id):
        """Look up the original file id of a file.

        :param file_id: New file id
        :return: Old file id if mapped, otherwise new file id
        """
        for x in self.map:
            if self.map[x] == file_id:
                return x
        return file_id

    def new_id(self, file_id):
        """Look up the new file id of a file.

        :param file_id: Old file id
        :return: New file id
        """
        try:
            return self.map[file_id]
        except KeyError:
            return file_id

    def get_file_sha1(self, path, file_id=None):
        """See Tree.get_file_sha1()."""
        return self.oldtree.get_file_sha1(path)

    def get_file_with_stat(self, path, file_id=None):
        """See Tree.get_file_with_stat()."""
        if getattr(self.oldtree, 'get_file_with_stat', None) is not None:
            return self.oldtree.get_file_with_stat(path=path)
        else:
            return (self.get_file(path), None)

    def get_file(self, path, file_id=None):
        """See Tree.get_file()."""
        return self.oldtree.get_file(path)

    def is_executable(self, path, file_id=None):
        """See Tree.is_executable()."""
        return self.oldtree.is_executable(path)

    def has_filename(self, filename):
        """See Tree.has_filename()."""
        return self.oldtree.has_filename(filename)

    def path_content_summary(self, path):
        """See Tree.path_content_summary()."""
        return self.oldtree.path_content_summary(path)

    def map_ie(self, ie):
        """Fix the references to old file ids in an inventory entry.

        :param ie: Inventory entry to map
        :return: New inventory entry
        """
        new_ie = ie.copy()
        new_ie.file_id = self.new_id(new_ie.file_id)
        new_ie.parent_id = self.new_id(new_ie.parent_id)
        return new_ie

    def iter_entries_by_dir(self):
        """See Tree.iter_entries_by_dir."""
        for path, ie in self.oldtree.iter_entries_by_dir():
            yield (path, self.map_ie(ie))

    def path2id(self, path):
        file_id = self.oldtree.path2id(path)
        if file_id is None:
            return None
        return self.new_id(file_id)

    def id2path(self, file_id, recurse='down'):
        return self.oldtree.id2path(self.old_id(file_id=file_id), recurse=recurse)