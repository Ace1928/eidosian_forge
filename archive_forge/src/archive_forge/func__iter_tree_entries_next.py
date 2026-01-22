import os
import os.path
import posixpath
import stat
from .compat import Collection, Iterable, string_types, unicode
def _iter_tree_entries_next(root_full, dir_rel, memo, on_error, follow_links):
    """
	Scan the directory for all descendant files.

	*root_full* (:class:`str`) the absolute path to the root directory.

	*dir_rel* (:class:`str`) the path to the directory to scan relative to
	*root_full*.

	*memo* (:class:`dict`) keeps track of ancestor directories
	encountered. Maps each ancestor real path (:class:`str`) to relative
	path (:class:`str`).

	*on_error* (:class:`~collections.abc.Callable` or :data:`None`)
	optionally is the error handler for file-system exceptions.

	*follow_links* (:class:`bool`) is whether to walk symbolic links that
	resolve to directories.

	Yields each entry (:class:`.TreeEntry`).
	"""
    dir_full = os.path.join(root_full, dir_rel)
    dir_real = os.path.realpath(dir_full)
    if dir_real not in memo:
        memo[dir_real] = dir_rel
    else:
        raise RecursionError(real_path=dir_real, first_path=memo[dir_real], second_path=dir_rel)
    for node_name in os.listdir(dir_full):
        node_rel = os.path.join(dir_rel, node_name)
        node_full = os.path.join(root_full, node_rel)
        try:
            node_lstat = os.lstat(node_full)
        except OSError as e:
            if on_error is not None:
                on_error(e)
            continue
        if stat.S_ISLNK(node_lstat.st_mode):
            is_link = True
            try:
                node_stat = os.stat(node_full)
            except OSError as e:
                if on_error is not None:
                    on_error(e)
                continue
        else:
            is_link = False
            node_stat = node_lstat
        if stat.S_ISDIR(node_stat.st_mode) and (follow_links or not is_link):
            yield TreeEntry(node_name, node_rel, node_lstat, node_stat)
            for entry in _iter_tree_entries_next(root_full, node_rel, memo, on_error, follow_links):
                yield entry
        elif stat.S_ISREG(node_stat.st_mode) or is_link:
            yield TreeEntry(node_name, node_rel, node_lstat, node_stat)
    del memo[dir_real]