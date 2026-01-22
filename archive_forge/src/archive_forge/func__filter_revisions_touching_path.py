import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _filter_revisions_touching_path(branch, path, view_revisions, include_merges=True):
    """Return the list of revision ids which touch a given path.

    The function filters view_revisions and returns a subset.
    This includes the revisions which directly change the path,
    and the revisions which merge these changes. So if the
    revision graph is::

        A-.
        |\\ \\
        B C E
        |/ /
        D |
        |\\|
        | F
        |/
        G

    And 'C' changes a file, then both C and D will be returned. F will not be
    returned even though it brings the changes to C into the branch starting
    with E. (Note that if we were using F as the tip instead of G, then we
    would see C, D, F.)

    This will also be restricted based on a subset of the mainline.

    :param branch: The branch where we can get text revision information.

    :param path: Filter out revisions that do not touch path.

    :param view_revisions: A list of (revision_id, dotted_revno, merge_depth)
        tuples. This is the list of revisions which will be filtered. It is
        assumed that view_revisions is in merge_sort order (i.e. newest
        revision first ).

    :param include_merges: include merge revisions in the result or not

    :return: A list of (revision_id, dotted_revno, merge_depth) tuples.
    """
    graph = branch.repository.get_file_graph()
    start_tree = branch.repository.revision_tree(view_revisions[0][0])
    file_id = start_tree.path2id(path)
    get_parent_map = graph.get_parent_map
    text_keys = [(file_id, rev_id) for rev_id, revno, depth in view_revisions]
    next_keys = None
    modified_text_revisions = set()
    chunk_size = 1000
    for start in range(0, len(text_keys), chunk_size):
        next_keys = text_keys[start:start + chunk_size]
        modified_text_revisions.update([k[1] for k in get_parent_map(next_keys)])
    del text_keys, next_keys
    result = []
    current_merge_stack = [None]
    for info in view_revisions:
        rev_id, revno, depth = info
        if depth == len(current_merge_stack):
            current_merge_stack.append(info)
        else:
            del current_merge_stack[depth + 1:]
            current_merge_stack[-1] = info
        if rev_id in modified_text_revisions:
            for idx in range(len(current_merge_stack)):
                node = current_merge_stack[idx]
                if node is not None:
                    if include_merges or node[2] == 0:
                        result.append(node)
                        current_merge_stack[idx] = None
    return result