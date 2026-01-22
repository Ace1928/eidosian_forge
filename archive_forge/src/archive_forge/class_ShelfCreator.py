import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
class ShelfCreator:
    """Create a transform to shelve objects and its inverse."""

    def __init__(self, work_tree, target_tree, file_list=None):
        """Constructor.

        :param work_tree: The working tree to apply changes to. This is not
            required to be locked - a tree_write lock will be taken out.
        :param target_tree: The tree to make the working tree more similar to.
            This is not required to be locked - a read_lock will be taken out.
        :param file_list: The files to make more similar to the target.
        """
        self.work_tree = work_tree
        self.work_transform = work_tree.transform()
        try:
            self.target_tree = target_tree
            self.shelf_transform = self.target_tree.preview_transform()
            try:
                self.renames = {}
                self.creation = {}
                self.deletion = {}
                self.iter_changes = work_tree.iter_changes(self.target_tree, specific_files=file_list)
            except:
                self.shelf_transform.finalize()
                raise
        except:
            self.work_transform.finalize()
            raise

    def iter_shelvable(self):
        """Iterable of tuples describing shelvable changes.

        As well as generating the tuples, this updates several members.
        Tuples may be::

           ('add file', file_id, work_kind, work_path)
           ('delete file', file_id, target_kind, target_path)
           ('rename', file_id, target_path, work_path)
           ('change kind', file_id, target_kind, work_kind, target_path)
           ('modify text', file_id)
           ('modify target', file_id, target_target, work_target)
        """
        for change in self.iter_changes:
            if change.kind[0] is None and change.name[1] == '':
                continue
            if change.kind[1] is None and change.name[0] == '':
                continue
            if change.kind[0] is None or change.versioned[0] is False:
                self.creation[change.file_id] = (change.kind[1], change.name[1], change.parent_id[1], change.versioned)
                yield ('add file', change.file_id, change.kind[1], change.path[1])
            elif change.kind[1] is None or change.versioned[0] is False:
                self.deletion[change.file_id] = (change.kind[0], change.name[0], change.parent_id[0], change.versioned)
                yield ('delete file', change.file_id, change.kind[0], change.path[0])
            else:
                if change.name[0] != change.name[1] or change.parent_id[0] != change.parent_id[1]:
                    self.renames[change.file_id] = (change.name, change.parent_id)
                    yield (('rename', change.file_id) + change.path)
                if change.kind[0] != change.kind[1]:
                    yield ('change kind', change.file_id, change.kind[0], change.kind[1], change.path[0])
                elif change.kind[0] == 'symlink':
                    t_target = self.target_tree.get_symlink_target(change.path[0])
                    w_target = self.work_tree.get_symlink_target(change.path[1])
                    yield ('modify target', change.file_id, change.path[0], t_target, w_target)
                elif change.changed_content:
                    yield ('modify text', change.file_id)

    def shelve_change(self, change):
        """Shelve a change in the iter_shelvable format."""
        if change[0] == 'rename':
            self.shelve_rename(change[1])
        elif change[0] == 'delete file':
            self.shelve_deletion(change[1])
        elif change[0] == 'add file':
            self.shelve_creation(change[1])
        elif change[0] in ('change kind', 'modify text'):
            self.shelve_content_change(change[1])
        elif change[0] == 'modify target':
            self.shelve_modify_target(change[1])
        else:
            raise ValueError('Unknown change kind: "%s"' % change[0])

    def shelve_all(self):
        """Shelve all changes.

        :return: ``True`` if changes were shelved, otherwise ``False``.
        """
        change = None
        for change in self.iter_shelvable():
            self.shelve_change(change)
        return change is not None

    def shelve_rename(self, file_id):
        """Shelve a file rename.

        :param file_id: The file id of the file to shelve the renaming of.
        """
        names, parents = self.renames[file_id]
        w_trans_id = self.work_transform.trans_id_file_id(file_id)
        work_parent = self.work_transform.trans_id_file_id(parents[0])
        self.work_transform.adjust_path(names[0], work_parent, w_trans_id)
        s_trans_id = self.shelf_transform.trans_id_file_id(file_id)
        shelf_parent = self.shelf_transform.trans_id_file_id(parents[1])
        self.shelf_transform.adjust_path(names[1], shelf_parent, s_trans_id)

    def shelve_modify_target(self, file_id):
        """Shelve a change of symlink target.

        :param file_id: The file id of the symlink which changed target.
        :param new_target: The target that the symlink should have due
            to shelving.
        """
        new_path = self.target_tree.id2path(file_id)
        new_target = self.target_tree.get_symlink_target(new_path)
        w_trans_id = self.work_transform.trans_id_file_id(file_id)
        self.work_transform.delete_contents(w_trans_id)
        self.work_transform.create_symlink(new_target, w_trans_id)
        old_path = self.work_tree.id2path(file_id)
        old_target = self.work_tree.get_symlink_target(old_path)
        s_trans_id = self.shelf_transform.trans_id_file_id(file_id)
        self.shelf_transform.delete_contents(s_trans_id)
        self.shelf_transform.create_symlink(old_target, s_trans_id)

    def shelve_lines(self, file_id, new_lines):
        """Shelve text changes to a file, using provided lines.

        :param file_id: The file id of the file to shelve the text of.
        :param new_lines: The lines that the file should have due to shelving.
        """
        w_trans_id = self.work_transform.trans_id_file_id(file_id)
        self.work_transform.delete_contents(w_trans_id)
        self.work_transform.create_file(new_lines, w_trans_id)
        s_trans_id = self.shelf_transform.trans_id_file_id(file_id)
        self.shelf_transform.delete_contents(s_trans_id)
        inverse_lines = self._inverse_lines(new_lines, file_id)
        self.shelf_transform.create_file(inverse_lines, s_trans_id)

    @staticmethod
    def _content_from_tree(tt, tree, file_id):
        trans_id = tt.trans_id_file_id(file_id)
        tt.delete_contents(trans_id)
        transform.create_from_tree(tt, trans_id, tree, tree.id2path(file_id))

    def shelve_content_change(self, file_id):
        """Shelve a kind change or binary file content change.

        :param file_id: The file id of the file to shelve the content change
            of.
        """
        self._content_from_tree(self.work_transform, self.target_tree, file_id)
        self._content_from_tree(self.shelf_transform, self.work_tree, file_id)

    def shelve_creation(self, file_id):
        """Shelve creation of a file.

        This handles content and inventory id.
        :param file_id: The file_id of the file to shelve creation of.
        """
        kind, name, parent, versioned = self.creation[file_id]
        version = not versioned[0]
        self._shelve_creation(self.work_tree, file_id, self.work_transform, self.shelf_transform, kind, name, parent, version)

    def shelve_deletion(self, file_id):
        """Shelve deletion of a file.

        This handles content and inventory id.
        :param file_id: The file_id of the file to shelve deletion of.
        """
        kind, name, parent, versioned = self.deletion[file_id]
        existing_path = self.target_tree.id2path(file_id)
        if not self.work_tree.has_filename(existing_path):
            existing_path = None
        version = not versioned[1]
        self._shelve_creation(self.target_tree, file_id, self.shelf_transform, self.work_transform, kind, name, parent, version, existing_path=existing_path)

    def _shelve_creation(self, tree, file_id, from_transform, to_transform, kind, name, parent, version, existing_path=None):
        w_trans_id = from_transform.trans_id_file_id(file_id)
        if parent is not None and kind is not None:
            from_transform.delete_contents(w_trans_id)
        from_transform.unversion_file(w_trans_id)
        if existing_path is not None:
            s_trans_id = to_transform.trans_id_tree_path(existing_path)
        else:
            s_trans_id = to_transform.trans_id_file_id(file_id)
        if parent is not None:
            s_parent_id = to_transform.trans_id_file_id(parent)
            to_transform.adjust_path(name, s_parent_id, s_trans_id)
            if existing_path is None:
                if kind is None:
                    to_transform.create_file([b''], s_trans_id)
                else:
                    transform.create_from_tree(to_transform, s_trans_id, tree, tree.id2path(file_id))
        if version:
            to_transform.version_file(s_trans_id, file_id=file_id)

    def _inverse_lines(self, new_lines, file_id):
        """Produce a version with only those changes removed from new_lines."""
        target_path = self.target_tree.id2path(file_id)
        target_lines = self.target_tree.get_file_lines(target_path)
        work_path = self.work_tree.id2path(file_id)
        work_lines = self.work_tree.get_file_lines(work_path)
        import patiencediff
        from merge3 import Merge3
        return Merge3(new_lines, target_lines, work_lines, sequence_matcher=patiencediff.PatienceSequenceMatcher).merge_lines()

    def finalize(self):
        """Release all resources used by this ShelfCreator."""
        self.work_transform.finalize()
        self.shelf_transform.finalize()

    def transform(self):
        """Shelve changes from working tree."""
        self.work_transform.apply()

    @staticmethod
    def metadata_record(serializer, revision_id, message=None):
        metadata = {b'revision_id': revision_id}
        if message is not None:
            metadata[b'message'] = message.encode('utf-8')
        return serializer.bytes_record(bencode.bencode(metadata), ((b'metadata',),))

    def write_shelf(self, shelf_file, message=None):
        """Serialize the shelved changes to a file.

        :param shelf_file: A file-like object to write the shelf to.
        :param message: An optional message describing the shelved changes.
        :return: the filename of the written file.
        """
        transform.resolve_conflicts(self.shelf_transform)
        revision_id = self.target_tree.get_revision_id()
        return self._write_shelf(shelf_file, self.shelf_transform, revision_id, message)

    @classmethod
    def _write_shelf(cls, shelf_file, transform, revision_id, message=None):
        serializer = pack.ContainerSerialiser()
        shelf_file.write(serializer.begin())
        metadata = cls.metadata_record(serializer, revision_id, message)
        shelf_file.write(metadata)
        for bytes in transform.serialize(serializer):
            shelf_file.write(bytes)
        shelf_file.write(serializer.end())