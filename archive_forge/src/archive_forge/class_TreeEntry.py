import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
class TreeEntry(object):
    """
	The :class:`.TreeEntry` class contains information about a file-system
	entry.
	"""
    __slots__ = ('_lstat', 'name', 'path', '_stat')

    def __init__(self, name, path, lstat, stat):
        """
		Initialize the :class:`.TreeEntry` instance.

		*name* (:class:`str`) is the base name of the entry.

		*path* (:class:`str`) is the relative path of the entry.

		*lstat* (:class:`~os.stat_result`) is the stat result of the direct
		entry.

		*stat* (:class:`~os.stat_result`) is the stat result of the entry,
		potentially linked.
		"""
        self._lstat = lstat
        '\n\t\t*_lstat* (:class:`~os.stat_result`) is the stat result of the direct\n\t\tentry.\n\t\t'
        self.name = name
        '\n\t\t*name* (:class:`str`) is the base name of the entry.\n\t\t'
        self.path = path
        '\n\t\t*path* (:class:`str`) is the path of the entry.\n\t\t'
        self._stat = stat
        '\n\t\t*_stat* (:class:`~os.stat_result`) is the stat result of the linked\n\t\tentry.\n\t\t'

    def is_dir(self, follow_links=None):
        """
		Get whether the entry is a directory.

		*follow_links* (:class:`bool` or :data:`None`) is whether to follow
		symbolic links. If this is :data:`True`, a symlink to a directory
		will result in :data:`True`. Default is :data:`None` for :data:`True`.

		Returns whether the entry is a directory (:class:`bool`).
		"""
        if follow_links is None:
            follow_links = True
        node_stat = self._stat if follow_links else self._lstat
        return stat.S_ISDIR(node_stat.st_mode)

    def is_file(self, follow_links=None):
        """
		Get whether the entry is a regular file.

		*follow_links* (:class:`bool` or :data:`None`) is whether to follow
		symbolic links. If this is :data:`True`, a symlink to a regular file
		will result in :data:`True`. Default is :data:`None` for :data:`True`.

		Returns whether the entry is a regular file (:class:`bool`).
		"""
        if follow_links is None:
            follow_links = True
        node_stat = self._stat if follow_links else self._lstat
        return stat.S_ISREG(node_stat.st_mode)

    def is_symlink(self):
        """
		Returns whether the entry is a symbolic link (:class:`bool`).
		"""
        return stat.S_ISLNK(self._lstat.st_mode)

    def stat(self, follow_links=None):
        """
		Get the cached stat result for the entry.

		*follow_links* (:class:`bool` or :data:`None`) is whether to follow
		symbolic links. If this is :data:`True`, the stat result of the
		linked file will be returned. Default is :data:`None` for :data:`True`.

		Returns that stat result (:class:`~os.stat_result`).
		"""
        if follow_links is None:
            follow_links = True
        return self._stat if follow_links else self._lstat