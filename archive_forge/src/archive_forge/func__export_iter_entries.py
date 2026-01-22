import errno
import os
import sys
import time
from . import archive, errors, osutils, trace
def _export_iter_entries(tree, subdir, skip_special=True, recurse_nested=False):
    """Iter the entries for tree suitable for exporting.

    :param tree: A tree object.
    :param subdir: None or the path of an entry to start exporting from.
    :param skip_special: Whether to skip .bzr files.
    :return: iterator over tuples with final path, tree path and inventory
        entry for each entry to export
    """
    if subdir == '':
        subdir = None
    if subdir is not None:
        subdir = subdir.rstrip('/')
    entries = tree.iter_entries_by_dir(recurse_nested=recurse_nested)
    for path, entry in entries:
        if path == '':
            continue
        if skip_special and tree.is_special_path(path):
            continue
        if path == subdir:
            if entry.kind == 'directory':
                continue
            final_path = entry.name
        elif subdir is not None:
            if path.startswith(subdir + '/'):
                final_path = path[len(subdir) + 1:]
            else:
                continue
        else:
            final_path = path
        if not tree.has_filename(path):
            continue
        yield (final_path, path, entry)