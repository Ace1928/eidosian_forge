from io import BytesIO
import os
import os.path as osp
from pathlib import Path
from stat import (
import subprocess
from git.cmd import handle_process_output, safer_popen
from git.compat import defenc, force_bytes, force_text, safe_decode
from git.exc import HookExecutionError, UnmergedEntriesError
from git.objects.fun import (
from git.util import IndexFileSHA1Writer, finalize_process
from gitdb.base import IStream
from gitdb.typ import str_tree_type
from .typ import BaseIndexEntry, IndexEntry, CE_NAMEMASK, CE_STAGESHIFT
from .util import pack, unpack
from typing import Dict, IO, List, Sequence, TYPE_CHECKING, Tuple, Type, Union, cast
from git.types import PathLike
def aggressive_tree_merge(odb: 'GitCmdObjectDB', tree_shas: Sequence[bytes]) -> List[BaseIndexEntry]:
    """
    :return: List of BaseIndexEntries representing the aggressive merge of the given
        trees. All valid entries are on stage 0, whereas the conflicting ones are left
        on stage 1, 2 or 3, whereas stage 1 corresponds to the common ancestor tree,
        2 to our tree and 3 to 'their' tree.

    :param tree_shas: 1, 2 or 3 trees as identified by their binary 20 byte shas.
        If 1 or two, the entries will effectively correspond to the last given tree.
        If 3 are given, a 3 way merge is performed.
    """
    out: List[BaseIndexEntry] = []
    if len(tree_shas) in (1, 2):
        for entry in traverse_tree_recursive(odb, tree_shas[-1], ''):
            out.append(_tree_entry_to_baseindexentry(entry, 0))
        return out
    if len(tree_shas) > 3:
        raise ValueError('Cannot handle %i trees at once' % len(tree_shas))
    for base, ours, theirs in traverse_trees_recursive(odb, tree_shas, ''):
        if base is not None:
            if ours is not None:
                if theirs is not None:
                    if base[0] != ours[0] and base[0] != theirs[0] and (ours[0] != theirs[0]) or (base[1] != ours[1] and base[1] != theirs[1] and (ours[1] != theirs[1])):
                        out.append(_tree_entry_to_baseindexentry(base, 1))
                        out.append(_tree_entry_to_baseindexentry(ours, 2))
                        out.append(_tree_entry_to_baseindexentry(theirs, 3))
                    elif base[0] != ours[0] or base[1] != ours[1]:
                        out.append(_tree_entry_to_baseindexentry(ours, 0))
                    else:
                        out.append(_tree_entry_to_baseindexentry(theirs, 0))
                elif ours[0] != base[0] or ours[1] != base[1]:
                    out.append(_tree_entry_to_baseindexentry(base, 1))
                    out.append(_tree_entry_to_baseindexentry(ours, 2))
            elif theirs is None:
                pass
            elif theirs[0] != base[0] or theirs[1] != base[1]:
                out.append(_tree_entry_to_baseindexentry(base, 1))
                out.append(_tree_entry_to_baseindexentry(theirs, 3))
        elif ours is None:
            assert theirs is not None
            out.append(_tree_entry_to_baseindexentry(theirs, 0))
        elif theirs is None:
            out.append(_tree_entry_to_baseindexentry(ours, 0))
        elif ours[0] != theirs[0] or ours[1] != theirs[1]:
            out.append(_tree_entry_to_baseindexentry(ours, 2))
            out.append(_tree_entry_to_baseindexentry(theirs, 3))
        else:
            out.append(_tree_entry_to_baseindexentry(ours, 0))
    return out