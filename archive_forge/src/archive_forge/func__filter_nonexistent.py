import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
def _filter_nonexistent(orig_paths, old_tree, new_tree):
    """Convert orig_paths to two sorted lists and return them.

    The first is orig_paths paths minus the items in the second list,
    and the second list is paths that are not in either inventory or
    tree (they don't qualify if they exist in the tree's inventory, or
    if they exist in the tree but are not versioned.)

    If either of the two lists is empty, return it as an empty list.

    This can be used by operations such as brz status that can accept
    unknown or ignored files.
    """
    mutter('check paths: %r', orig_paths)
    if not orig_paths:
        return (orig_paths, [])
    s = old_tree.filter_unversioned_files(orig_paths)
    s = new_tree.filter_unversioned_files(s)
    nonexistent = [path for path in s if not new_tree.has_filename(path)]
    remaining = [path for path in orig_paths if path not in nonexistent]
    return (sorted(remaining), sorted(nonexistent))