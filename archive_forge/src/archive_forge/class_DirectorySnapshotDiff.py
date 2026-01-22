import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
class DirectorySnapshotDiff(object):
    """
    Compares two directory snapshots and creates an object that represents
    the difference between the two snapshots.

    :param ref:
        The reference directory snapshot.
    :type ref:
        :class:`DirectorySnapshot`
    :param snapshot:
        The directory snapshot which will be compared
        with the reference snapshot.
    :type snapshot:
        :class:`DirectorySnapshot`
    """

    def __init__(self, ref, snapshot):
        created = snapshot.paths - ref.paths
        deleted = ref.paths - snapshot.paths
        for path in ref.paths & snapshot.paths:
            if ref.inode(path) != snapshot.inode(path):
                created.add(path)
                deleted.add(path)
        moved = set()
        for path in set(deleted):
            inode = ref.inode(path)
            new_path = snapshot.path(inode)
            if new_path:
                deleted.remove(path)
                moved.add((path, new_path))
        for path in set(created):
            inode = snapshot.inode(path)
            old_path = ref.path(inode)
            if old_path:
                created.remove(path)
                moved.add((old_path, path))
        modified = set()
        for path in ref.paths & snapshot.paths:
            if ref.inode(path) == snapshot.inode(path):
                if ref.mtime(path) != snapshot.mtime(path):
                    modified.add(path)
        for old_path, new_path in moved:
            if ref.mtime(old_path) != snapshot.mtime(new_path):
                modified.add(old_path)
        self._dirs_created = [path for path in created if snapshot.isdir(path)]
        self._dirs_deleted = [path for path in deleted if ref.isdir(path)]
        self._dirs_modified = [path for path in modified if ref.isdir(path)]
        self._dirs_moved = [(frm, to) for frm, to in moved if ref.isdir(frm)]
        self._files_created = list(created - set(self._dirs_created))
        self._files_deleted = list(deleted - set(self._dirs_deleted))
        self._files_modified = list(modified - set(self._dirs_modified))
        self._files_moved = list(moved - set(self._dirs_moved))

    @property
    def files_created(self):
        """List of files that were created."""
        return self._files_created

    @property
    def files_deleted(self):
        """List of files that were deleted."""
        return self._files_deleted

    @property
    def files_modified(self):
        """List of files that were modified."""
        return self._files_modified

    @property
    def files_moved(self):
        """
        List of files that were moved.

        Each event is a two-tuple the first item of which is the path
        that has been renamed to the second item in the tuple.
        """
        return self._files_moved

    @property
    def dirs_modified(self):
        """
        List of directories that were modified.
        """
        return self._dirs_modified

    @property
    def dirs_moved(self):
        """
        List of directories that were moved.

        Each event is a two-tuple the first item of which is the path
        that has been renamed to the second item in the tuple.
        """
        return self._dirs_moved

    @property
    def dirs_deleted(self):
        """
        List of directories that were deleted.
        """
        return self._dirs_deleted

    @property
    def dirs_created(self):
        """
        List of directories that were created.
        """
        return self._dirs_created