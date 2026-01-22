from . import commit, controldir, errors, revision
def _flush_pending(self, tree, pending):
    """Flush the pending actions in 'pending', i.e. apply them to tree."""
    for path, file_id in pending.to_add_directories:
        if path == '':
            if tree.has_filename(path) and path in pending.to_unversion_paths:
                pending.to_unversion_paths.discard(path)
            if file_id is not None:
                tree.add([path], ['directory'], ids=[file_id])
            else:
                tree.add([path], ['directory'])
        elif file_id is not None:
            tree.mkdir(path, file_id)
        else:
            tree.mkdir(path)
    for from_relpath, to_relpath in pending.to_rename:
        tree.rename_one(from_relpath, to_relpath)
    if pending.to_unversion_paths:
        tree.unversion(pending.to_unversion_paths)
    if tree.supports_file_ids:
        tree.add(pending.to_add_files, pending.to_add_kinds, pending.to_add_file_ids)
    else:
        tree.add(pending.to_add_files, pending.to_add_kinds)
    for path, content in pending.new_contents.items():
        tree.put_file_bytes_non_atomic(path, content)