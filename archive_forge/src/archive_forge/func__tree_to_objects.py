import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def _tree_to_objects(tree, parent_trees, idmap, unusual_modes, dummy_file_name=None, add_cache_entry=None):
    """Iterate over the objects that were introduced in a revision.

    :param idmap: id map
    :param parent_trees: Parent revision trees
    :param unusual_modes: Unusual file modes dictionary
    :param dummy_file_name: File name to use for dummy files
        in empty directories. None to skip empty directories
    :return: Yields (path, object, ie) entries
    """
    dirty_dirs = set()
    new_blobs = []
    shamap = {}
    try:
        base_tree = parent_trees[0]
        other_parent_trees = parent_trees[1:]
    except IndexError:
        base_tree = tree._repository.revision_tree(NULL_REVISION)
        other_parent_trees = []

    def find_unchanged_parent_ie(path, kind, other, parent_trees):
        for ptree in parent_trees:
            intertree = InterTree.get(ptree, tree)
            ppath = intertree.find_source_path(path)
            if ppath is not None:
                pkind = ptree.kind(ppath)
                if kind == 'file':
                    if pkind == 'file' and ptree.get_file_sha1(ppath) == other:
                        return (ptree.path2id(ppath), ptree.get_file_revision(ppath))
                if kind == 'symlink':
                    if pkind == 'symlink' and ptree.get_symlink_target(ppath) == other:
                        return (ptree.path2id(ppath), ptree.get_file_revision(ppath))
        raise KeyError
    for change in tree.iter_changes(base_tree):
        if change.name[1] in BANNED_FILENAMES:
            continue
        if change.kind[1] == 'file':
            sha1 = tree.get_file_sha1(change.path[1])
            blob_id = None
            try:
                pfile_id, prevision = find_unchanged_parent_ie(change.path[1], change.kind[1], sha1, other_parent_trees)
            except KeyError:
                pass
            else:
                try:
                    blob_id = idmap.lookup_blob_id(pfile_id, prevision)
                except KeyError:
                    if not change.changed_content:
                        blob = Blob()
                        blob.data = tree.get_file_text(change.path[1])
                        blob_id = blob.id
            if blob_id is None:
                new_blobs.append((change.path[1], change.file_id))
            else:
                shamap[change.path[1]] = blob_id
                if add_cache_entry is not None:
                    add_cache_entry(('blob', blob_id), (change.file_id, tree.get_file_revision(change.path[1])), change.path[1])
        elif change.kind[1] == 'symlink':
            target = tree.get_symlink_target(change.path[1])
            blob = symlink_to_blob(target)
            shamap[change.path[1]] = blob.id
            if add_cache_entry is not None:
                add_cache_entry(blob, (change.file_id, tree.get_file_revision(change.path[1])), change.path[1])
            try:
                find_unchanged_parent_ie(change.path[1], change.kind[1], target, other_parent_trees)
            except KeyError:
                if change.changed_content:
                    yield (change.path[1], blob, (change.file_id, tree.get_file_revision(change.path[1])))
        elif change.kind[1] is None:
            shamap[change.path[1]] = None
        elif change.kind[1] != 'directory':
            raise AssertionError(change.kind[1])
        for p in change.path:
            if p is None:
                continue
            dirty_dirs.add(osutils.dirname(p))
    for (path, file_id), chunks in tree.iter_files_bytes([(path, (path, file_id)) for path, file_id in new_blobs]):
        obj = Blob()
        obj.chunked = list(chunks)
        if add_cache_entry is not None:
            add_cache_entry(obj, (file_id, tree.get_file_revision(path)), path)
        yield (path, obj, (file_id, tree.get_file_revision(path)))
        shamap[path] = obj.id
    for path in unusual_modes:
        dirty_dirs.add(posixpath.dirname(path))
    for dir in list(dirty_dirs):
        for parent in osutils.parent_directories(dir):
            if parent in dirty_dirs:
                break
            dirty_dirs.add(parent)
    if dirty_dirs:
        dirty_dirs.add('')

    def ie_to_hexsha(path, ie):
        try:
            return shamap[path]
        except KeyError:
            pass
        if ie.kind == 'file':
            try:
                return idmap.lookup_blob_id(ie.file_id, ie.revision)
            except KeyError:
                blob = Blob()
                blob.data = tree.get_file_text(path)
                if add_cache_entry is not None:
                    add_cache_entry(blob, (ie.file_id, ie.revision), path)
                return blob.id
        elif ie.kind == 'symlink':
            try:
                return idmap.lookup_blob_id(ie.file_id, ie.revision)
            except KeyError:
                target = tree.get_symlink_target(path)
                blob = symlink_to_blob(target)
                if add_cache_entry is not None:
                    add_cache_entry(blob, (ie.file_id, ie.revision), path)
                return blob.id
        elif ie.kind == 'directory':
            ret = directory_to_tree(path, ie.children.values(), ie_to_hexsha, unusual_modes, dummy_file_name, ie.parent_id is None)
            if ret is None:
                return ret
            return ret.id
        else:
            raise AssertionError
    for path in sorted(dirty_dirs, reverse=True):
        if not tree.has_filename(path):
            continue
        if tree.kind(path) != 'directory':
            continue
        obj = directory_to_tree(path, tree.iter_child_entries(path), ie_to_hexsha, unusual_modes, dummy_file_name, path == '')
        if obj is not None:
            file_id = tree.path2id(path)
            if add_cache_entry is not None:
                add_cache_entry(obj, (file_id, tree.get_revision_id()), path)
            yield (path, obj, (file_id, tree.get_revision_id()))
            shamap[path] = obj.id