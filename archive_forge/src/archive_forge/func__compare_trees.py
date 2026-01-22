from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def _compare_trees(old_tree, new_tree, want_unchanged, specific_files, include_root, extra_trees=None, require_versioned=False, want_unversioned=False):
    """Worker function that implements Tree.changes_from."""
    delta = TreeDelta()
    for change in new_tree.iter_changes(old_tree, want_unchanged, specific_files, extra_trees=extra_trees, require_versioned=require_versioned, want_unversioned=want_unversioned):
        if change.versioned == (False, False):
            delta.unversioned.append(change)
            continue
        if not include_root and (None, None) == change.parent_id:
            continue
        fully_present = tuple((change.versioned[x] and change.kind[x] is not None for x in range(2)))
        if fully_present[0] != fully_present[1]:
            if fully_present[1] is True:
                delta.added.append(change)
            elif change.kind[0] == 'symlink' and (not new_tree.supports_symlinks()):
                trace.warning('Ignoring "%s" as symlinks are not supported on this filesystem.' % (change.path[0],))
            else:
                delta.removed.append(change)
        elif fully_present[0] is False:
            delta.missing.append(change)
        elif change.name[0] != change.name[1] or change.parent_id[0] != change.parent_id[1]:
            if change.copied:
                delta.copied.append(change)
            else:
                delta.renamed.append(change)
        elif change.kind[0] != change.kind[1]:
            delta.kind_changed.append(change)
        elif change.changed_content or change.executable[0] != change.executable[1]:
            delta.modified.append(change)
        else:
            delta.unchanged.append(change)

    def change_key(change):
        if change.path[0] is None:
            path = change.path[1]
        else:
            path = change.path[0]
        return (path, change.file_id)
    delta.removed.sort(key=change_key)
    delta.added.sort(key=change_key)
    delta.renamed.sort(key=change_key)
    delta.copied.sort(key=change_key)
    delta.missing.sort(key=change_key)
    delta.modified.sort(key=change_key)
    delta.unchanged.sort(key=change_key)
    delta.unversioned.sort(key=change_key)
    return delta